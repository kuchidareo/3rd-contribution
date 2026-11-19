"""Cache frozen-tokenizer codes and the full training-set level-zero prior."""
from pathlib import Path

import numpy as np
import torch

from baseline_data.dataloader import TokenDataset, sequential_batches
from training.runtime import (cli, fingerprint, json_write, load_config, load_vqvae,
                              seed_all, sha256)


@torch.no_grad()
def encode(cfg, batch_size=32):
    seed_all(cfg.seed)
    device = torch.device(cfg.device)
    vq = load_vqvae(cfg, device)
    out = Path(cfg.paths.codes)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Code cache exists; use a new paths.codes directory')
    out.mkdir(parents=True, exist_ok=True)
    levels = list(vq.vq.levels)
    counts = np.zeros(vq.vq.vqvae_vocab_sizes[0], dtype=np.int64)
    for split in ['train', 'validation', 'test']:
        data = TokenDataset(cfg.paths.data, split)
        dtype = np.uint16 if max(vq.vq.vqvae_vocab_sizes) <= 65536 else np.int32
        target = np.lib.format.open_memmap(out / f'{split}.npy', mode='w+', dtype=dtype,
                                          shape=(len(data), sum(levels)))
        offset = 0
        # Code assignment deliberately uses FP32 to avoid precision-dependent targets.
        for batch in sequential_batches(data, batch_size):
            codes = vq.encode(batch['tokens'].to(device)).cpu().numpy()
            target[offset:offset + len(codes)] = codes
            if split == 'train':
                counts += np.bincount(codes[:, :levels[0]].ravel(), minlength=len(counts))
            offset += len(codes)
        target.flush()
        print(f'{split}: cached {offset} code rows', flush=True)
    np.save(out / 'level0_counts.npy', counts)
    json_write({'schema': 1, 'levels': levels, 'assignment_precision': 'fp32',
                'tokenizer_sha256': sha256(cfg.paths.tokenizer_checkpoint),
                'tokenizer_config_hash': fingerprint(dict_config(cfg.upstream.tokenizer.vqvae)),
                'data_manifest_sha256': sha256(Path(cfg.paths.data) / 'manifest.json'),
                'split_sha256': {s: sha256(out / f'{s}.npy') for s in ['train', 'validation', 'test']},
                'level0_counts_sha256': sha256(out / 'level0_counts.npy')}, out / 'manifest.json')


def dict_config(cfg):
    from omegaconf import OmegaConf
    obj = OmegaConf.to_container(cfg, resolve=True)
    obj.pop('checkpoint_path', None)
    return obj


def main():
    ap = cli(__doc__)
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()
    encode(load_config(args.config, args.set), args.batch_size)


if __name__ == '__main__':
    main()
