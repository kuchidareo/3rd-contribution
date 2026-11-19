"""Single-device training with accumulation, atomic checkpoints and deterministic resume."""
import json
import math
import shutil
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from baseline_data.dataloader import TokenDataset, sampled_batch, sequential_batches
from models.autoregressive import AutoregressiveTransformer
from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE
from training.objectives import ar_loss, ncm_loss, tokenizer_loss
from training.amp import optimizer_update
from training.stopping import EarlyStopping, compatible_resume
from training.runtime import (atomic_save, autocast_context, cli, environment, fingerprint,
                              json_write, load_config, load_vqvae, parameter_counts,
                              restore_rng, rng_state, seed_all, sha256, tokenizer_config)


def verify_data(cfg, with_codes):
    root = Path(cfg.paths.data)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['sequence_length'] != cfg.ar.context_length or manifest['vocab_size'] != cfg.ar.vocab_size:
        raise ValueError('Prepared dataset and model vocabulary/length do not match')
    for split in ['train', 'validation', 'test']:
        if sha256(root / f'{split}.npy') != manifest['split_sha256'][split]:
            raise ValueError(f'Token data changed: {split}')
    for key in ['bpe_vocab', 'bpe_merges']:
        name = cfg.upstream.checkpoints[key]
        if sha256(Path(cfg.paths.artifacts) / name) != manifest['assets_sha256'][name]:
            raise ValueError('Text tokenizer assets differ from prepared data')
    if with_codes:
        cache = Path(cfg.paths.codes)
        meta = json.loads((cache / 'manifest.json').read_text())
        if meta['data_manifest_sha256'] != sha256(root / 'manifest.json'):
            raise ValueError('Code cache belongs to different data')
        if meta['tokenizer_sha256'] != sha256(cfg.paths.tokenizer_checkpoint):
            raise ValueError('Code cache belongs to different tokenizer weights')
        from baseline_data.encode_codes import dict_config
        if meta['tokenizer_config_hash'] != fingerprint(dict_config(cfg.upstream.tokenizer.vqvae)):
            raise ValueError('Code cache tokenizer configuration mismatch')
        for split in ['train', 'validation', 'test']:
            if sha256(cache / f'{split}.npy') != meta['split_sha256'][split]:
                raise ValueError(f'Code cache changed: {split}')
        if sha256(cache / 'level0_counts.npy') != meta['level0_counts_sha256']:
            raise ValueError('Level-zero prior changed')
    return manifest


def learning_rate(step, recipe):
    # step is the zero-based update index. Explicit cosine choice: unpublished upstream scheduler.
    if step < recipe.warmup_steps:
        return recipe.lr * (step + 1) / max(recipe.warmup_steps, 1)
    progress = (step - recipe.warmup_steps) / max(recipe.steps - recipe.warmup_steps - 1, 1)
    return recipe.min_lr + 0.5 * (recipe.lr - recipe.min_lr) * (1 + math.cos(math.pi * min(progress, 1)))


def build_model(kind, cfg, device):
    if kind == 'ar':
        return AutoregressiveTransformer(cfg.ar).to(device), None
    if kind == 'tokenizer':
        return VQVAE(**tokenizer_config(cfg).vqvae).to(device), None
    frozen = load_vqvae(cfg, device)
    return NextConceptTransformer(vqvae_local=frozen, config=cfg.upstream.model).to(device), frozen


def batch_loss(kind, model, frozen, batch, eos_id, cfg, training):
    if kind == 'ar':
        return ar_loss(model, batch['tokens'], eos_id), {}
    if kind == 'tokenizer':
        return tokenizer_loss(model, batch['tokens'])
    return ncm_loss(model, frozen, batch['codes'], eos_id,
                    corruption_start=cfg.training.ncm_corruption_start,
                    corruption_end=cfg.training.ncm_corruption_end,
                    alpha=cfg.training.ncm_loss_alpha, training=training), {}


@torch.no_grad()
def validate(kind, model, frozen, data, cfg, micro_batch_size, eos_id):
    state = rng_state()
    model.eval()
    total, count, metrics = 0.0, 0, {}
    try:
        for batch in sequential_batches(data, micro_batch_size, cfg.training.validation_rows):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            with autocast_context(cfg.device, cfg.precision):
                loss, extra = batch_loss(kind, model, frozen, batch, eos_id, cfg, False)
            n = len(batch['tokens'])
            total += float(loss) * n
            count += n
            for k, value in extra.items():
                metrics[k] = metrics.get(k, 0.0) + float(value) * n
        return {'loss': total / count, **{k: value / count for k, value in metrics.items()}}
    finally:
        restore_rng(state)
        model.train()


def run(kind, args):
    cfg = load_config(args.config, args.set)
    device = torch.device(cfg.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --set device=cpu --set precision=fp32 for a smoke test')
    seed_all(cfg.seed)
    manifest = verify_data(cfg, kind == 'ncm')
    recipe = cfg.training.tokenizer if kind == 'tokenizer' else cfg.training.generator
    micro = int(recipe.micro_batch_size)
    if micro < 1 or recipe.global_batch_size % micro:
        raise ValueError('global_batch_size must be a positive multiple of micro_batch_size')
    for field in ['validation_every', 'checkpoint_every', 'log_every', 'validation_rows']:
        if int(cfg.training[field]) < 1:
            raise ValueError(f'training.{field} must be positive')
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError('--max-steps must be positive')
    accumulation = int(recipe.global_batch_size) // micro
    if recipe.steps < 1 or accumulation < 1:
        raise ValueError('steps and accumulation must be positive')
    out = Path(args.out or f'outputs/{kind}')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'latest.pt').exists() and not args.resume:
        raise FileExistsError('Output already has a checkpoint; use --resume or a new --out')
    train_data = TokenDataset(cfg.paths.data, 'train', cfg.paths.codes if kind == 'ncm' else None)
    valid_data = TokenDataset(cfg.paths.data, 'validation', cfg.paths.codes if kind == 'ncm' else None)
    model, frozen = build_model(kind, cfg, device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=recipe.lr,
                                  betas=tuple(cfg.training.betas), weight_decay=recipe.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and cfg.precision == 'fp16'))
    # Portable resume permits path relocation but forbids silently changing data or training policy.
    signature = {'kind': kind, 'seed': cfg.seed, 'precision': cfg.precision,
                 'recipe': OmegaConf.to_container(recipe, resolve=True),
                 'training': OmegaConf.to_container(cfg.training, resolve=True),
                 'ar': OmegaConf.to_container(cfg.ar, resolve=True),
                 'upstream': OmegaConf.to_container(cfg.upstream, resolve=True),
                 'data_manifest_sha256': sha256(Path(cfg.paths.data) / 'manifest.json')}
    if kind == 'ncm':
        signature['code_manifest_sha256'] = sha256(Path(cfg.paths.codes) / 'manifest.json')
    signature_hash = fingerprint(signature)
    prior = None
    if kind == 'ncm':
        counts = torch.from_numpy(np.load(Path(cfg.paths.codes) / 'level0_counts.npy')).double()
        if counts.sum() <= 0:
            raise ValueError('Empty level-zero counts')
        prior = (counts / counts.sum()).float()
    max_amp_retries = int(cfg.training.get('amp_max_retries', 16))
    if max_amp_retries < 0:
        raise ValueError('amp_max_retries must be nonnegative')
    policy = dict(enabled=True, patience=5, min_delta=0.001, min_steps=500)
    policy.update(OmegaConf.to_container(cfg.training.get('early_stopping', OmegaConf.create({})), resolve=True))
    policy['min_steps'] = max(int(policy['min_steps']), int(recipe.warmup_steps))
    selection = EarlyStopping(**policy)
    step = 0
    overflow_retries_total = 0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        if not compatible_resume(signature, ck):
            raise ValueError('Resume configuration/data differs from saved checkpoint')
        model.load_state_dict(ck['model'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        if kind == 'tokenizer' and 'fine_drop_step' in ck:
            model.vq._fd_step.fill_(ck['fine_drop_step'])
        step = ck['step']
        selection = EarlyStopping.restore(ck.get('early_stopping'), **policy)
        overflow_retries_total = int(ck.get('amp_overflow_retries', 0))
        restore_rng(ck['rng'])
    if args.resume and selection.best_step is not None:
        best_source = out / 'best.pt' if (out / 'best.pt').exists() else Path(args.resume).parent / 'best.pt'
        if not best_source.exists():
            raise FileNotFoundError('Resume requires best.pt alongside latest.pt to preserve the historical best model. Attach both files.')
        best_header = torch.load(best_source, map_location='cpu', weights_only=False, mmap=True)
        if best_header.get('kind') != kind or best_header['step'] != selection.best_step or not compatible_resume(signature, best_header):
            raise ValueError('The restored best.pt does not match the resume state')
        del best_header
        if best_source.resolve() != (out / 'best.pt').resolve():
            shutil.copyfile(best_source, out / 'best.pt')
    if cfg.training.compile:
        # Compile forward only: preserve upstream checkpoint keys and model API.
        model.forward = torch.compile(model.forward)
    model.train()
    metadata = {'kind': kind, 'config': OmegaConf.to_container(cfg, resolve=True),
                'environment': environment(), **parameter_counts(model),
                'data_manifest': manifest, 'signature_hash': signature_hash,
                'generator_text_positions_per_update': recipe.global_batch_size * cfg.ar.context_length,
                'tokenizer_parameters_separate': parameter_counts(frozen) if frozen is not None else None,
                'tokenizer_checkpoint_sha256': sha256(cfg.paths.tokenizer_checkpoint) if frozen is not None else None,
                'tokenizer_training_text_positions': frozen.source_training_text_positions if frozen is not None else None,
                'tokenizer_training_provenance': frozen.source_training_metadata if frozen is not None else None,
                'amp_policy': {'max_retries_per_update': max_amp_retries,
                               'retry_same_batch_and_rng': True, 'count_only_applied_updates': True}}
    json_write(metadata, out / 'run.json')
    stop = min(recipe.steps, args.max_steps) if args.max_steps else int(recipe.steps)
    log_path = out / 'metrics.jsonl'

    stop_reason = 'running'

    def save(filename='latest.pt'):
        ck = {'schema': 1, 'kind': kind, 'step': step, 'model': model.state_dict(),
              'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'rng': rng_state(),
              'signature_hash': signature_hash, 'metadata': metadata,
              'early_stopping': selection.state_dict(), 'stop_reason': stop_reason,
              'text_positions_seen': step * recipe.global_batch_size * cfg.ar.context_length,
              'amp_overflow_retries': overflow_retries_total,
              'attempted_text_positions': (step + overflow_retries_total) * recipe.global_batch_size * cfg.ar.context_length}
        if kind == 'tokenizer':
            ck['fine_drop_step'] = int(model.vq._fd_step)
        if prior is not None:
            ck['inference'] = {'level0_unigram': prior}
        atomic_save(ck, out / filename)

    # Older checkpoints have no selection history: establish a validation baseline
    # from their actual weights instead of pretending it is the historical best.
    if args.resume and selection.best_step is None:
        baseline = validate(kind, model, frozen, valid_data, cfg, micro, manifest['eos_id'])
        selection.observe(baseline['loss'], step)
        save('best.pt')
        print(json.dumps({'event': 'resume_validation_baseline', 'step': step, 'validation_clean': baseline}), flush=True)
    if step >= stop or selection.stopped:
        stop_reason = 'early_stopping' if selection.stopped else ('budget_complete' if step >= recipe.steps else 'session_limit')
        save()
    while step < stop and not selection.stopped:
        started = time.perf_counter()
        # A retry repeats identical data/dropout/corruption at a lower scale.
        retry_rng = rng_state() if scaler.is_enabled() else None
        vq_snapshot = None
        fine_drop_step = None
        if scaler.is_enabled() and kind == 'tokenizer':
            # Tokenizer forward mutates EMA codebooks, hit counters and fine-drop step.
            vq_snapshot = {k: v.detach().clone() for k, v in model.vq.state_dict().items()}
            fine_drop_step = model.vq._fd_step.detach().clone()
        retries = 0
        while True:
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for acc in range(accumulation):
                batch = sampled_batch(train_data, micro, cfg.seed, step * accumulation + acc, device)
                with autocast_context(device, cfg.precision):
                    loss, _ = batch_loss(kind, model, frozen, batch, manifest['eos_id'], cfg, True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite forward loss at update {step + 1}; reducing the gradient scale cannot repair forward overflow')
                scaler.scale(loss / accumulation).backward()
                train_loss += float(loss.detach()) / accumulation
            lr = learning_rate(step, recipe)
            for group in optimizer.param_groups:
                group['lr'] = lr
            update = optimizer_update(
                optimizer, scaler,
                lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip))
            if update.applied:
                norm = update.grad_norm
                break
            retries += 1
            overflow_retries_total += 1
            restore_rng(retry_rng)
            if vq_snapshot is not None:
                model.vq.load_state_dict(vq_snapshot)
                model.vq._fd_step.copy_(fine_drop_step)
            event = {'event': 'fp16_overflow_retry', 'update': step + 1, 'retry': retries,
                     'scale_before': update.scale_before, 'scale_after': update.scale_after}
            with open(out / 'amp_events.jsonl', 'a') as stream:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
            print(json.dumps(event), flush=True)
            if retries > max_amp_retries:
                save()  # Preserves the last applied update and restored RNG/EMA state.
                raise FloatingPointError(f'FP16 update {step + 1} exceeded {max_amp_retries} retries; checkpoint saved without advancing the update count')
        del vq_snapshot
        step += 1
        row = {'step': step, 'train_loss': train_loss, 'lr': lr, 'grad_norm': float(norm),
               'update_seconds': time.perf_counter() - started,
               'amp_retries_this_update': retries, 'amp_overflow_retries_total': overflow_retries_total,
               'loss_scale': float(scaler.get_scale()),
               'attempted_text_positions': (step + overflow_retries_total) * recipe.global_batch_size * cfg.ar.context_length,
               'text_positions_seen': step * recipe.global_batch_size * cfg.ar.context_length}
        improved = False
        if step % cfg.training.validation_every == 0 or step == stop:
            row['validation_clean'] = validate(kind, model, frozen, valid_data, cfg, micro, manifest['eos_id'])
            improved = selection.observe(row['validation_clean']['loss'], step)
            row['selection'] = selection.state_dict()
        if selection.stopped:
            stop_reason = 'early_stopping'
        elif step == stop:
            stop_reason = 'budget_complete' if step == recipe.steps else 'session_limit'
        with open(log_path, 'a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if step % cfg.training.log_every == 0 or step == stop or selection.stopped:
            print(json.dumps(row), flush=True)
        if improved:
            save('best.pt')
        if step % cfg.training.checkpoint_every == 0 or step == stop or selection.stopped:
            save()
    json_write({'kind': kind, 'step': step, 'target_steps': int(recipe.steps),
                'stop_reason': stop_reason, 'best_step': selection.best_step, 'best_loss': selection.best_loss,
                'full_budget_completed': step == recipe.steps}, out / 'training_status.json')
    print(f'{kind}: {stop_reason}; best update {selection.best_step}, validation loss {selection.best_loss}', flush=True)
    print(f'{kind}: update {step}; checkpoint {out / "latest.pt"}', flush=True)


def main(kind):
    ap = cli(f'Train {kind} on the common TinyStories token rows')
    ap.add_argument('--out')
    ap.add_argument('--resume', help='Path to latest.pt (includes optimizer, sampler position and RNG)')
    ap.add_argument('--max-steps', type=int, help='Stop after this absolute update; preserve full LR horizon')
    run(kind, ap.parse_args())
