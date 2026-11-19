"""Generate and benchmark one model per process; no reference-model download here."""
import json
from contextlib import nullcontext
from pathlib import Path

from omegaconf import OmegaConf
import torch

from evaluation.benchmark import benchmark
from models.autoregressive import AutoregressiveTransformer
from models.next_concept import NextConceptTransformer
from training.runtime import (autocast_context, cli, environment, json_write, load_config,
                              load_vqvae, parameter_counts, seed_all, sha256, text_tokenizer)


def decode_ids(ids, tokenizer):
    ids = list(ids)
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    if eos in ids:
        ids = ids[:ids.index(eos)]
    return tokenizer.decode([i for i in ids if i not in (eos, pad)], skip_special_tokens=True)


def load_generator(kind, checkpoint, cfg, device):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'kind' in ck and ck['kind'] != kind:
        raise ValueError('Checkpoint model type does not match --model')
    frozen, prior = None, None
    if kind == 'ar':
        if 'metadata' not in ck:
            raise ValueError('AR checkpoint must include its architecture metadata')
        saved_ar = OmegaConf.create(ck['metadata']['config']['ar'])
        if saved_ar != cfg.ar:
            raise ValueError('AR architecture differs from the checkpoint')
        model = AutoregressiveTransformer(saved_ar).to(device)
    else:
        frozen = load_vqvae(cfg, device)
        model = NextConceptTransformer(vqvae_local=frozen, config=cfg.upstream.model).to(device)
        prior = ck['inference']['level0_unigram'].float().cpu()
        if prior.ndim != 1 or prior.numel() != frozen.vq.vqvae_vocab_sizes[0] or not torch.isfinite(prior).all() or (prior < 0).any() or prior.sum() <= 0:
            raise ValueError('Invalid level-zero prior')
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    return model, frozen, prior, ck.get('metadata'), ck.get('text_positions_seen'), {
        'step': ck.get('step'), 'stop_reason': ck.get('stop_reason'),
        'early_stopping': ck.get('early_stopping')}


def main():
    ap = cli(__doc__)
    ap.add_argument('--model', choices=['ar', 'ncm'], required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if cfg.evaluation.samples_per_seed < 1 or cfg.evaluation.batch_size < 1:
        raise ValueError('Sample count and evaluation batch size must be positive')
    if not cfg.evaluation.seeds or len(set(cfg.evaluation.seeds)) != len(cfg.evaluation.seeds):
        raise ValueError('Evaluation seeds must be nonempty and distinct')
    out = Path(args.out)
    if (out / 'generation.json').exists():
        raise FileExistsError('Evaluation output already exists; use a new --out')
    device = torch.device(cfg.device)
    seed_all(cfg.seed)
    tk = text_tokenizer(cfg)
    model, frozen, prior, training, text_positions, selection = load_generator(args.model, args.checkpoint, cfg, device)
    if training:
        recorded = training['data_manifest']['assets_sha256']
        for key in ['bpe_vocab', 'bpe_merges']:
            name = cfg.upstream.checkpoints[key]
            if sha256(Path(cfg.paths.artifacts) / name) != recorded[name]:
                raise ValueError('Evaluation BPE assets differ from training')
        if args.model == 'ncm':
            # The portable provenance hash is also recorded in each training run below.
            recorded_hash = training.get('tokenizer_checkpoint_sha256')
            if recorded_hash and recorded_hash != sha256(cfg.paths.tokenizer_checkpoint):
                raise ValueError('NCM tokenizer differs from training')
    sampling = {key: cfg.sampling[key] for key in ['temperature', 'top_k', 'top_p']}
    length = int(cfg.ar.context_length)

    @torch.inference_mode()
    def generate(batch_size):
        with autocast_context(device, cfg.precision):
            if args.model == 'ar':
                return model.generate(batch_size, length, tk.eos_token_id,
                                      use_kv_cache=cfg.sampling.ar_kv_cache, **sampling)
            l0 = torch.multinomial(prior, batch_size * frozen.vq.levels[0], replacement=True)
            l0 = l0.view(batch_size, frozen.vq.levels[0]).to(device)
            prefix = torch.full((batch_size, 1), tk.eos_token_id, device=device, dtype=torch.long)
            codes = model.generate(l0, prefix, **sampling)
            # Same final reconstruction as official idx_to_zhats(...)[-1], without intermediates.
            with (torch.autocast(device_type=device.type, enabled=False)
                  if model.selective_fp32 else nullcontext()):
                z = frozen.vq.idx_to_z_q(codes)
            return frozen.decoder(z).argmax(-1)

    result = benchmark(generate, model, frozen.decoder if frozen is not None else None,
                       device, int(cfg.evaluation.batch_size), length,
                       int(cfg.evaluation.warmup_batches), int(cfg.evaluation.benchmark_repeats))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'generation.json').exists():
        raise FileExistsError('Evaluation output already exists; use a new --out')
    with open(out / 'samples.jsonl', 'w') as stream:
        for seed in cfg.evaluation.seeds:
            seed_all(int(seed))
            n = int(cfg.evaluation.samples_per_seed)
            for start in range(0, n, int(cfg.evaluation.batch_size)):
                ids = generate(min(int(cfg.evaluation.batch_size), n - start)).cpu().tolist()
                for offset, row in enumerate(ids):
                    text = decode_ids(row, tk)
                    stream.write(json.dumps({'seed': int(seed), 'index': start + offset, 'text': text,
                                             'raw_ids': row, 'entropy_ids': tk.encode(text)}, ensure_ascii=False) + '\n')
    payload = {'schema': 1, 'model': args.model, 'checkpoint_sha256': sha256(args.checkpoint),
               'tokenizer_checkpoint_sha256': sha256(cfg.paths.tokenizer_checkpoint) if frozen is not None else None,
               'assets_sha256': {cfg.upstream.checkpoints[k]: sha256(Path(cfg.paths.artifacts) / cfg.upstream.checkpoints[k])
                                 for k in ['bpe_vocab', 'bpe_merges']},
               'environment': environment(), 'config': OmegaConf.to_container(cfg, resolve=True),
               'generator': parameter_counts(model),
               'tokenizer': parameter_counts(frozen) if frozen is not None else None,
               'training_text_positions': text_positions,
               'checkpoint_selection': selection,
               'tokenizer_training_text_positions': frozen.source_training_text_positions if frozen is not None else 0,
               'training_provenance': training,
               'protocol': {'length': length, 'batch_size': int(cfg.evaluation.batch_size),
                            'seeds': list(cfg.evaluation.seeds), 'samples_per_seed': int(cfg.evaluation.samples_per_seed),
                            'precision': cfg.precision, 'sampling': sampling,
                            'cut_at_eos_for_quality': True, 'fixed_length_for_timing': True,
                            'ar_kv_cache': bool(cfg.sampling.ar_kv_cache), 'ncm_kv_cache': False},
               'benchmark': result, 'samples_sha256': sha256(out / 'samples.jsonl')}
    json_write(payload, out / 'generation.json')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
