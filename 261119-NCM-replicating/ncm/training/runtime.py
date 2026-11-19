import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
import torch

from utils.experiment_io import fingerprint, json_write, sha256

ROOT = Path(__file__).resolve().parents[1]


def load_config(path, overrides=()):
    cfg = OmegaConf.load(path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    official = OmegaConf.load(ROOT / cfg.official_config)
    # Full upstream config remains the source of tokenizer and NCM architecture.
    official = OmegaConf.merge(official, cfg.get('upstream_overrides', {}))
    cfg.upstream = official
    cfg.ar.context_length = int(official.tokenizer.vqvae.context_length)
    cfg.ar.vocab_size = int(official.tokenizer.vqvae.in_vocab_size)
    return cfg


def cli(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', default='config/baseline/tinystories.yaml')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    return parser


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device, precision):
    if precision == 'fp32':
        return nullcontext()
    if precision == 'bf16':
        if torch.device(device).type == 'cuda' and not torch.cuda.is_bf16_supported():
            raise ValueError('This GPU lacks BF16 support. Use --set precision=fp16 (recorded deviation).')
        return torch.autocast(torch.device(device).type, dtype=torch.bfloat16)
    if precision == 'fp16' and torch.device(device).type == 'cuda':
        return torch.autocast('cuda', dtype=torch.float16)
    raise ValueError('precision must be fp32, bf16, or fp16 (CUDA only)')


def environment():
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'git_revision': revision, 'git_dirty': dirty}


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda'] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def tokenizer_config(cfg, checkpoint=None):
    tc = OmegaConf.create(OmegaConf.to_container(cfg.upstream.tokenizer, resolve=True))
    root = Path(cfg.paths.artifacts)
    tc.pre_tokenizer.vocab_file_path = str(root / cfg.upstream.checkpoints.bpe_vocab)
    tc.pre_tokenizer.merges_file_path = str(root / cfg.upstream.checkpoints.bpe_merges)
    tc.vqvae.checkpoint_path = str(checkpoint or cfg.paths.tokenizer_checkpoint)
    return tc


def text_tokenizer(cfg):
    from tokenization.bpe_tokenizer import BPETokenizer
    return BPETokenizer.from_config(tokenizer_config(cfg).pre_tokenizer)


def load_vqvae(cfg, device, checkpoint=None):
    from models.vqvae import VQVAE
    tc = tokenizer_config(cfg, checkpoint)
    model = VQVAE(**tc.vqvae).to(device)
    checkpoint = torch.load(tc.vqvae.checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.source_training_metadata = checkpoint.get('metadata')
    model.source_training_text_positions = checkpoint.get('text_positions_seen')
    return model.eval().requires_grad_(False)


def parameter_counts(model):
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
