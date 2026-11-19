"""Memory-mapped token rows; deterministic replacement sampling supports exact resume."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, root, split, codes_dir=None):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.tokens = np.load(self.root / f'{split}.npy', mmap_mode='r')
        if self.tokens.ndim != 2 or not len(self.tokens):
            raise ValueError(f'{split}: expected nonempty [rows, length] token array')
        self.codes = None
        if codes_dir:
            self.codes = np.load(Path(codes_dir) / f'{split}.npy', mmap_mode='r')
            if len(self.codes) != len(self.tokens):
                raise ValueError('Token/code row count mismatch')

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        result = {'tokens': torch.from_numpy(np.array(self.tokens[index], dtype=np.int64))}
        if self.codes is not None:
            result['codes'] = torch.from_numpy(np.array(self.codes[index], dtype=np.int64))
        return result


def sampled_batch(dataset, size, seed, micro_step, device):
    # A private RNG never consumes the model/dropout RNG; no sampler cursor to lose.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(micro_step)]))
    indexes = rng.integers(0, len(dataset), size=int(size))
    rows = [dataset[int(i)] for i in indexes]
    return {k: torch.stack([r[k] for r in rows]).to(device) for k in rows[0]}


def sequential_batches(dataset, size, limit=None):
    end = len(dataset) if limit is None else min(len(dataset), int(limit))
    for start in range(0, end, size):
        rows = [dataset[i] for i in range(start, min(start + size, end))]
        yield {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
