"""Explicit experiment profiles; pure Python so profile policy can be tested offline."""
from copy import deepcopy


PAPER_GENERATOR = {
    'steps': 50000,
    'global_batch_size': 2048,
    'lr': 0.0003,
    'min_lr': 0.00003,
    'warmup_steps': 500,
    'weight_decay': 0.1,
}

# These are reconstruction choices, not values established by the public training release.
TRAINING_ASSUMPTIONS = {
    'learning_rate_curve': 'linear warmup followed by cosine decay',
    'conditioning_corruption': 'uniform code replacement, linear per-level rates 0.9 to 0.1',
    'ncm_loss_alpha': 0.0,
    'ar_embedding_head_tying': True,
    'optimizer_parameter_groups': 'weight decay applied to all trainable parameters',
    'training_windows': 'saved nonoverlapping source-record chunks, right padded, no new EOS',
    'training_row_sampler': 'sampling with replacement',
    'padding_loss': 'included in training loss; not inferred from token-budget accounting',
    'heldout_split': 'saved deterministic division of official validation records',
    'tokenizer': 'reuse official pretrained tokenizer; tokenizer training is not rerun',
    'ar_architecture': 'local causal adaptation of upstream blocks; exact released AR is unavailable',
    'timing': 'local synchronized benchmark, separate from published quality conditions',
}


def configure_profile(config, official, profile, *, bf16_supported, micro_batch=2,
                      pilot_steps=100, pilot_global_batch=32, model_overrides=None, precision=None):
    if profile not in {'pilot', 'paper'}:
        raise ValueError("PROFILE must be 'pilot' or 'paper'")
    if precision not in {None, 'bf16', 'fp16', 'fp32'}:
        raise ValueError('PRECISION must be bf16, fp16, or fp32')
    if micro_batch < 1:
        raise ValueError('MICRO_BATCH must be positive')
    cfg = deepcopy(config)
    overrides = model_overrides or {}
    recipe = cfg['training']['generator']
    recipe['micro_batch_size'] = int(micro_batch)
    if profile == 'paper':
        selected_precision = precision or 'bf16'
        if selected_precision not in {'bf16', 'fp16'}:
            raise ValueError("PROFILE='paper' supports BF16 or an explicit FP16 hardware adaptation")
        if selected_precision == 'bf16' and not bf16_supported:
            raise RuntimeError("BF16 requires a BF16-capable GPU. On T4 explicitly set PRECISION='fp16'; "
                               "the full training budget will be preserved and the precision deviation recorded.")
        if overrides or cfg.get('upstream_overrides'):
            raise ValueError("PROFILE='paper' uses the unmodified official NCM architecture; remove model overrides")
        recipe.update(PAPER_GENERATOR)
        cfg['seed'] = 42
        cfg['precision'] = selected_precision
        cfg['training'].update({'betas': [0.9, 0.95], 'grad_clip': 1.0, 'compile': True,
                                'ncm_corruption_start': 0.9, 'ncm_corruption_end': 0.1,
                                'ncm_loss_alpha': 0.0})
        m = official['model']
        # Match the backbone settings, while recording that AR embedding/head choices are ours.
        cfg['ar'].update({k: m[k] for k in ['n_layer', 'n_head', 'n_embd', 'bias', 'dropout',
                                           'use_qk_norm', 'rope_base']})
        cfg['ar']['tie_embeddings'] = True
        ev = official['eval']
        cfg['evaluation'].update({'seeds': list(ev['seeds']), 'samples_per_seed': ev['n_samples'],
                                  'batch_size': ev['batch_size'], 'ref_model': ev['ref_model'],
                                  'scoring_batch_size': 32, 'scoring_max_length': 1024,
                                  'warmup_batches': 2, 'benchmark_repeats': 5})
        cfg['sampling'].update({k: official['generation'][k] for k in ['temperature', 'top_k', 'top_p']})
        cfg['sampling']['ar_kv_cache'] = True
    else:
        if pilot_steps < 1 or pilot_global_batch < 1:
            raise ValueError('Pilot steps and global batch must be positive')
        cfg['precision'] = precision or ('bf16' if bf16_supported else 'fp32')
        if cfg['precision'] == 'bf16' and not bf16_supported:
            raise RuntimeError('BF16 requires a BF16-capable GPU')
        cfg['training']['compile'] = False
        recipe.update({'steps': int(pilot_steps), 'global_batch_size': int(pilot_global_batch),
                       'warmup_steps': min(10, int(pilot_steps))})
        cfg['training'].update({'validation_every': 25, 'checkpoint_every': 25, 'validation_rows': 64})
        cfg['evaluation'].update({'seeds': [42], 'samples_per_seed': 32, 'batch_size': 4,
                                  'benchmark_repeats': 3})
        if overrides:
            cfg.setdefault('upstream_overrides', {})['model'] = deepcopy(overrides)
    if recipe['global_batch_size'] % micro_batch:
        raise ValueError('MICRO_BATCH must divide the selected global batch size')
    cfg['ncm_selective_fp32'] = cfg['precision'] == 'fp16'
    cfg['experiment'] = {
        'profile': profile,
        'status': 'published settings with explicitly reconstructed training details' if profile == 'paper' else 'pilot only',
        'full_paper_equivalence_verified': False,
        'training_assumptions': deepcopy(TRAINING_ASSUMPTIONS),
        'gradient_accumulation_steps': recipe['global_batch_size'] // micro_batch,
        'target_generator_text_positions': recipe['steps'] * recipe['global_batch_size'] *
                                          official['tokenizer']['vqvae']['context_length'],
        'completion_required_for_final_comparison': profile == 'paper',
        'ncm_precision_policy': ('FP16 transformer; FP32 frozen VQ conditioning and tied codebook heads'
                                 if cfg['ncm_selective_fp32'] else 'unchanged autocast policy'),
        'paper_precision_matched': cfg['precision'] == 'bf16',
        'hardware_deviations': (["FP16 with gradient scaling replaces the paper's BF16 precision"]
                                if profile == 'paper' and cfg['precision'] == 'fp16' else []),
    }
    return cfg


def check_checkpoint_progress(kind, checkpoint, profile, target_steps, *, precision='bf16'):
    """Validate headers before resuming or presenting a final paper comparison."""
    if checkpoint.get('kind') != kind:
        raise ValueError(f'{kind}: wrong checkpoint kind')
    if profile == 'paper':
        saved = checkpoint.get('metadata', {}).get('config', {})
        recipe = saved.get('training', {}).get('generator', {})
        if precision not in {'bf16', 'fp16'} or any(recipe.get(k) != v for k, v in PAPER_GENERATOR.items()) or saved.get('precision') != precision:
            raise ValueError(f'{kind}: cannot resume/evaluate a pilot or different training recipe as paper. '
                             'Use a fresh paper run or a matching paper checkpoint with the same precision.')
    step = checkpoint.get('step')
    if not isinstance(step, int) or not 0 <= step <= target_steps:
        raise ValueError(f'{kind}: invalid checkpoint update count {step!r}')
    return step
