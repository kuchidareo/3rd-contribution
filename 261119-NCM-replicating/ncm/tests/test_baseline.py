"""Synthetic regression tests. No dataset, checkpoint or scorer downloads."""
import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from baseline_data.dataloader import TokenDataset, sampled_batch
from baseline_data.prepare import BinaryRecords, prepare
from evaluation.benchmark import benchmark
from evaluation.compare import compare
from models.autoregressive import AutoregressiveTransformer
from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE
from training.objectives import ar_loss, corrupt_codes, ncm_loss, tokenizer_loss
from training.runtime import load_config, sha256
from training.train import run


def tiny_config(tmp_path):
    cfg = load_config('config/baseline/tinystories.yaml', [
        'device=cpu', 'precision=fp32', 'training.compile=false',
        'ar.n_layer=1', 'ar.n_head=2', 'ar.n_embd=16', 'ar.dropout=0.0',
        'upstream_overrides.tokenizer.vqvae.in_vocab_size=16',
        'upstream_overrides.tokenizer.vqvae.context_length=8',
        'upstream_overrides.tokenizer.vqvae.embed_dim=8',
        'upstream_overrides.tokenizer.vqvae.n_layers=2',
        'upstream_overrides.tokenizer.vqvae.pre_quant_groupnorm=2',
        'upstream_overrides.tokenizer.vqvae.vqvae_vocab_size=[8,8,8]',
        'upstream_overrides.tokenizer.vqvae.vector_quantizer_config.levels.manual_levels=[1,4,8]',
        'upstream_overrides.model.n_layer=1', 'upstream_overrides.model.n_head=2',
        'upstream_overrides.model.n_embd=16', 'upstream_overrides.model.dropout=0.0',
        'upstream_overrides.model.vocab_size=[8,8]', 'upstream_overrides.model.context_length=9',
    ])
    cfg.paths.data = str(tmp_path / 'data')
    cfg.paths.artifacts = str(tmp_path / 'assets')
    cfg.paths.codes = str(tmp_path / 'codes')
    cfg.paths.tokenizer_checkpoint = str(tmp_path / 'tokenizer.pt')
    return cfg


def synthetic_data(tmp_path, cfg):
    assets = Path(cfg.paths.artifacts)
    assets.mkdir()
    # Enough to test asset identity; tests do not instantiate text BPE.
    paths = [assets / cfg.upstream.checkpoints[k] for k in ['bpe_vocab', 'bpe_merges']]
    for p in paths:
        p.write_bytes(b'synthetic tokenizer asset')
    train = tmp_path / 'train.pickle'
    valid = tmp_path / 'valid.pickle'
    for p, rows in [(train, [[2, 3, 4, 0], [5, 6, 0]] * 4),
                    (valid, [[2, 3, 0], [4, 5, 0], [6, 7, 0], [8, 9, 0]])]:
        p.write_bytes(pickle.dumps(rows))
    return prepare(train, valid, cfg.paths.data, SimpleNamespace(pad_token_id=1, eos_token_id=0),
                   length=8, vocab_size=16, asset_paths=paths)


def test_ar_causality_and_cache(tmp_path):
    torch.manual_seed(7)
    cfg = tiny_config(tmp_path)
    model = AutoregressiveTransformer(cfg.ar).eval()
    a = torch.tensor([[0, 2, 3, 4, 5, 6, 7, 8]])
    b = a.clone()
    b[:, 4:] = 9
    torch.testing.assert_close(model(a)[:, :4], model(b)[:, :4])
    full = model(a)
    for block in model.blocks:
        block.attn.kv_caching(True)
    cached = torch.cat([model(a[:, i:i+1]) for i in range(8)], dim=1)
    for block in model.blocks:
        block.attn.kv_caching(False)
    torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-5)
    greedy = dict(temperature=0, top_k=0, top_p=1.0)
    torch.testing.assert_close(model.generate(2, 8, 0, use_kv_cache=True, **greedy),
                               model.generate(2, 8, 0, use_kv_cache=False, **greedy))


def test_ncm_objective_generation_and_decoder_parity(tmp_path):
    cfg = tiny_config(tmp_path)
    vq = VQVAE(**cfg.upstream.tokenizer.vqvae)
    tokens = torch.randint(2, 16, (2, 8))
    vq.train()
    loss, metrics = tokenizer_loss(vq, tokens)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in vq.encoder.parameters())
    vq.eval().requires_grad_(False)
    codes = vq.encode(tokens)
    original = codes.clone()
    model = NextConceptTransformer(vq, cfg.upstream.model)
    value = ncm_loss(model, vq, codes, 0, corruption_start=.9, corruption_end=.1, alpha=0, training=True)
    value.backward()
    torch.testing.assert_close(codes, original)
    assert all(p.grad is None or not p.requires_grad for p in vq.parameters())
    assert torch.isfinite(value)
    model.eval()
    generated = model.generate(codes[:, :1], torch.zeros(2, 1, dtype=torch.long), top_k=0, top_p=1)
    assert generated.shape == (2, 13)
    _, levels = vq.vq.idx_to_zhats(generated, return_zlevels=True)
    torch.testing.assert_close(vq.vq.idx_to_z_q(generated), levels[-1])
    assert vq.decoder(levels[-1]).shape == (2, 8, 16)
    noisy = corrupt_codes(codes, [1, 4, 8], [8, 8, 8], 1, 1)
    assert noisy.min() >= 0 and noisy.max() < 8
    torch.testing.assert_close(codes, original)


def test_data_split_and_sampling_resume(tmp_path):
    cfg = tiny_config(tmp_path)
    manifest = synthetic_data(tmp_path, cfg)
    assert manifest['rows'] == {'train': 8, 'validation': 2, 'test': 2}
    valid = np.load(Path(cfg.paths.data) / 'validation.npy')
    test = np.load(Path(cfg.paths.data) / 'test.npy')
    assert set(valid[:, 0]).isdisjoint(test[:, 0])
    ds = TokenDataset(cfg.paths.data, 'train')
    a = sampled_batch(ds, 4, 42, 17, 'cpu')
    b = sampled_batch(ds, 4, 42, 17, 'cpu')
    torch.testing.assert_close(a['tokens'], b['tokens'])


@pytest.mark.parametrize('kind', ['ar', 'tokenizer', 'ncm'])
def test_checkpoint_resume_exact(tmp_path, kind):
    cfg = tiny_config(tmp_path)
    synthetic_data(tmp_path, cfg)
    cfg.training.generator.steps = 2
    cfg.training.generator.global_batch_size = 4
    cfg.training.generator.micro_batch_size = 2
    cfg.training.generator.warmup_steps = 0
    cfg.training.validation_every = 1
    cfg.training.checkpoint_every = 1
    cfg.training.validation_rows = 2
    cfg.ar.dropout = .1
    cfg.upstream_overrides.model.dropout = .1
    cfg.training.tokenizer = OmegaConf.create(OmegaConf.to_container(cfg.training.generator))
    if kind == 'ncm':
        from baseline_data.encode_codes import encode
        frozen = VQVAE(**cfg.upstream.tokenizer.vqvae)
        torch.save({'model': frozen.state_dict()}, cfg.paths.tokenizer_checkpoint)
        encode(cfg, batch_size=2)
    # Save a config without runtime-expanded upstream; preserve synthetic overrides.
    file = tmp_path / 'config.yaml'
    OmegaConf.save(cfg, file)
    full, split = tmp_path / 'full', tmp_path / 'split'
    def train(out, max_steps=None, resume=None):
        run(kind, SimpleNamespace(config=str(file), set=[], out=str(out), max_steps=max_steps, resume=resume))
    train(full)
    train(split, 1)
    train(split, resume=str(split / 'latest.pt'))
    a = torch.load(full / 'latest.pt', weights_only=False)
    b = torch.load(split / 'latest.pt', weights_only=False)
    assert a['step'] == b['step'] == 2
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
    assert a['text_positions_seen'] == b['text_positions_seen']


def test_benchmark_counts_real_ar_forwards(tmp_path):
    cfg = tiny_config(tmp_path)
    model = AutoregressiveTransformer(cfg.ar).eval()
    result = benchmark(lambda size: model.generate(size, 8, 0, temperature=0), model, None,
                       'cpu', 2, 8, warmup=1, repeats=2)
    assert result['forward_passes_per_batch'] == [{'generator': 8, 'decoder': 0}] * 2
    assert result['tokens_per_second'] > 0
    assert result['peak_allocated_bytes'] is None


def test_paper_parameter_counts():
    # Meta tensors count the actual released architecture without allocating 0.5GB of weights.
    cfg = load_config('config/baseline/tinystories.yaml')
    with torch.device('meta'):
        ar = AutoregressiveTransformer(cfg.ar)
    ar_count = sum(p.numel() for p in ar.parameters())
    assert 88_000_000 < ar_count < 93_000_000


def test_comparison_rejects_mismatched_conditions():
    base = {'model': 'ar', 'protocol': dict(length=128, batch_size=16, seeds=[42], samples_per_seed=2,
            precision='fp32', sampling={}, cut_at_eos_for_quality=True, fixed_length_for_timing=True),
            'assets_sha256': {}, 'environment': dict(gpu=None, torch='test', cuda=None),
            'benchmark': dict(warmup_batches=1, repeats=2, forward_passes_per_batch=[dict(generator=128, decoder=0)],
                              median_batch_latency_seconds=1., tokens_per_second=256.,
                              peak_allocated_bytes=None, peak_reserved_bytes=None),
            'training_provenance': None, 'training_text_positions': None,
            'generator': {'parameters': 1}, 'tokenizer': None}
    other = copy.deepcopy(base)
    other['model'] = 'ncm'
    other['protocol']['batch_size'] = 8
    with pytest.raises(ValueError, match='protocol.batch_size'):
        compare(base, other)
    result = compare(base, other, allow_unmatched=True)
    assert not result['matched_conditions']


def test_binary_records_with_and_without_sentinel(tmp_path):
    path = tmp_path / 'tokens.bin'
    np.array([2, 3, 0, 4, 5, 6, 0], dtype=np.uint16).tofile(path)
    offsets = tmp_path / 'offsets.npy'
    for positions in [[0, 3], [0, 3, 7]]:
        np.save(offsets, np.array(positions, dtype=np.int64))
        data = BinaryRecords(path, offsets)
        assert len(data) == 2
        assert [row.tolist() for row in data] == [[2, 3, 0], [4, 5, 6, 0]]
    np.save(offsets, np.array([0, 8]))
    with pytest.raises(ValueError, match='Offsets'):
        BinaryRecords(path, offsets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for real FP16 GradScaler integration')
def test_real_cuda_scaler_rejects_overflow_then_applies_retry():
    from training.amp import optimizer_update
    parameter = torch.nn.Parameter(torch.tensor([1.0], device='cuda'))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    scaler.scale(parameter.square().sum()).backward()
    parameter.grad.fill_(float('inf'))
    before = parameter.detach().clone()
    rejected = optimizer_update(optimizer, scaler, lambda: torch.nn.utils.clip_grad_norm_([parameter], 1.))
    assert not rejected.applied
    assert scaler.get_scale() == 512.
    torch.testing.assert_close(parameter, before)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(parameter.square().sum()).backward()
    accepted = optimizer_update(optimizer, scaler, lambda: torch.nn.utils.clip_grad_norm_([parameter], 1.))
    assert accepted.applied
    assert parameter.item() < before.item()


def test_training_saves_best_and_latest_then_stops(tmp_path, monkeypatch):
    import training.train as trainer
    cfg = tiny_config(tmp_path)
    synthetic_data(tmp_path, cfg)
    cfg.training.generator.steps = 4
    cfg.training.generator.warmup_steps = 0
    cfg.training.generator.global_batch_size = 4
    cfg.training.generator.micro_batch_size = 2
    cfg.training.validation_every = 1
    cfg.training.checkpoint_every = 100
    cfg.training.early_stopping = dict(enabled=True, patience=1, min_delta=.001, min_steps=0)
    config = tmp_path / 'selection.yaml'
    OmegaConf.save(cfg, config)
    snapshots = []
    losses = iter([3., 2., 2.5])
    def validation(kind, model, *args):
        snapshots.append({key: value.detach().clone() for key, value in model.state_dict().items()})
        return {'loss': next(losses)}
    monkeypatch.setattr(trainer, 'validate', validation)
    out = tmp_path / 'selection-run'
    args = SimpleNamespace(config=str(config), set=[], out=str(out), max_steps=None, resume=None)
    trainer.run('ar', args)
    best = torch.load(out / 'best.pt', weights_only=False)
    latest = torch.load(out / 'latest.pt', weights_only=False)
    assert best['step'] == 2
    assert latest['step'] == 3
    assert latest['stop_reason'] == 'early_stopping'
    assert latest['early_stopping']['best_step'] == 2
    for key in best['model']:
        torch.testing.assert_close(best['model'][key], snapshots[1][key], rtol=0, atol=0)
        torch.testing.assert_close(latest['model'][key], snapshots[2][key], rtol=0, atol=0)
    args.resume = str(out / 'latest.pt')
    trainer.run('ar', args)
    assert torch.load(out / 'latest.pt', weights_only=False)['step'] == 3
    assert len(snapshots) == 3  # Resuming a stopped run does not silently restart training.
