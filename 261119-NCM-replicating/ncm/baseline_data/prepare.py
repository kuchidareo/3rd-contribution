"""Kaggle-side artifact acquisition and deterministic official-token data preparation."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from training.runtime import json_write, load_config, sha256, text_tokenizer

DATA_REVISION = '03000aa'  # public release; full resolved revision is recorded by download
MODEL_REVISION = '7100573'


def records(obj):
    if isinstance(obj, dict):
        if 'input_ids' not in obj:
            raise ValueError('Unknown pickle mapping; expected input_ids')
        obj = obj['input_ids']
    if hasattr(obj, 'tolist'):
        obj = obj.tolist()
    if not isinstance(obj, (tuple, list)) or not len(obj):
        raise ValueError('Expected nonempty token sequence or list of token sequences')
    if isinstance(obj[0], (int, np.integer)):
        return [obj], 'flat_stream'
    if isinstance(obj[0], dict):
        obj = [r['input_ids'] for r in obj]
    result = []
    for row in obj:
        row = row.tolist() if hasattr(row, 'tolist') else row
        if not isinstance(row, (tuple, list)) or any(not isinstance(x, (int, np.integer)) for x in row):
            raise ValueError('Expected integer token IDs in each record')
        result.append(row)
    return result, 'source_records'


class BinaryRecords:
    """Explicit uint16 flat binary plus document offsets; never unpickle the training corpus."""
    def __init__(self, path, offsets_path, dtype='uint16'):
        self.tokens = np.memmap(path, mode='r', dtype=np.dtype(dtype))
        offsets = np.load(offsets_path, mmap_mode='r')
        if offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer):
            raise ValueError('Expected a one-dimensional integer offset array')
        if len(offsets) < 1 or offsets[0] != 0 or offsets[-1] > len(self.tokens) or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError('Offsets must be sorted token indexes starting at zero within the binary')
        self.offsets = offsets
        self.count = len(offsets) - int(offsets[-1] == len(self.tokens))

    def __len__(self):
        return self.count

    def __iter__(self):
        for i in range(self.count):
            end = int(self.offsets[i + 1]) if i + 1 < len(self.offsets) else len(self.tokens)
            yield self.tokens[int(self.offsets[i]):end]


def chunks(rows, length, pad_id):
    for row in rows:
        for start in range(0, len(row), length):
            part = list(row[start:start + length])
            yield part + [pad_id] * (length - len(part))


def write_rows(rows, path, length, pad_id, vocab_size):
    count = sum((len(row) + length - 1) // length for row in rows)
    if count == 0:
        raise ValueError(f'Empty split: {path}')
    out = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint16, shape=(count, length))
    for index, row in enumerate(chunks(rows, length, pad_id)):
        if min(row) < 0 or max(row) >= vocab_size:
            raise ValueError('Source token IDs fall outside configured text vocabulary')
        out[index] = row
    out.flush()
    return count


def prepare(train_path, valid_path, out_dir, tokenizer, length=128, seed=42, vocab_size=8192,
            test_fraction=0.5, asset_paths=(), train_offsets=None, bin_dtype="uint16"):
    out_dir = Path(out_dir)
    if (out_dir / 'manifest.json').exists():
        raise FileExistsError('Prepared dataset already exists; use a new output directory')
    if not 0 < test_fraction < 1:
        raise ValueError('test_fraction must be between zero and one')
    if train_offsets is not None:
        train = BinaryRecords(train_path, train_offsets, bin_dtype)
        train_layout = f'flat_binary_{bin_dtype}_with_record_offsets'
    else:
        with open(train_path, 'rb') as stream:
            train, train_layout = records(pickle.load(stream))
    with open(valid_path, 'rb') as stream:
        heldout, valid_layout = records(pickle.load(stream))
    # Never divide chunks of the same source record across held-out sets.
    # A flat stream has no published document boundaries: split nonoverlapping windows.
    if valid_layout == 'flat_stream':
        heldout = list(chunks(heldout, length, tokenizer.pad_token_id))
    if len(heldout) < 2:
        raise ValueError('Need at least two held-out records/windows for validation and test')
    order = np.random.default_rng(seed).permutation(len(heldout))
    ntest = max(1, min(len(heldout) - 1, round(len(heldout) * test_fraction)))
    test_index = set(order[:ntest].tolist())
    validation = [r for i, r in enumerate(heldout) if i not in test_index]
    test = [r for i, r in enumerate(heldout) if i in test_index]
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in [('train', train), ('validation', validation), ('test', test)]:
        counts[name] = write_rows(rows, out_dir / f'{name}.npy', length, tokenizer.pad_token_id, vocab_size)
    manifest = {'schema': 1, 'sequence_length': length, 'vocab_size': vocab_size,
                'pad_id': tokenizer.pad_token_id, 'eos_id': tokenizer.eos_token_id,
                'seed': seed, 'test_fraction_of_official_valid': test_fraction,
                'train_layout': train_layout, 'valid_layout': valid_layout,
                'policy': 'source records chunked without overlap and right-padded; no added EOS; official valid split before chunking unless flat stream',
                'source_sha256': {Path(p).name: sha256(p) for p in [train_path, valid_path] + ([train_offsets] if train_offsets else [])},
                'assets_sha256': {Path(p).name: sha256(p) for p in asset_paths},
                'rows': counts, 'split_sha256': {s: sha256(out_dir / f'{s}.npy') for s in counts}}
    json_write(manifest, out_dir / 'manifest.json')
    return manifest


def download(destination, revision_models=MODEL_REVISION, revision_data=DATA_REVISION):
    # Called only by explicit CLI --download, intended for Kaggle.
    from huggingface_hub import HfApi, snapshot_download
    root = Path(destination)
    entries = []
    for repo, kind, revision, target, patterns in [
        ('hazyresearch/ncm-checkpoints', 'model', revision_models, root / 'artifacts', ['tinystories/*']),
        ('hazyresearch/ncm-datasets', 'dataset', revision_data, root / 'official-data',
         ['tinystories/train_bpe_8192.bin', 'tinystories/train_bpe_8192_offsets.npy',
          'tinystories/valid_bpe_8192.pickle', 'tinystories/bpe_*.pickle', 'tinystories/README.md'])]:
        resolved = HfApi().repo_info(repo, repo_type=kind, revision=revision).sha
        snapshot_download(repo, repo_type=kind, revision=resolved, allow_patterns=patterns, local_dir=target)
        entries.append({'repo': repo, 'revision': resolved, 'destination': str(target)})
    json_write(entries, root / 'downloads.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='config/baseline/tinystories.yaml')
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--download-root', default='.')
    ap.add_argument('--models-revision', default=MODEL_REVISION)
    ap.add_argument('--data-revision', default=DATA_REVISION)
    ap.add_argument('--train', default='official-data/tinystories/train_bpe_8192.bin')
    ap.add_argument('--train-format', choices=['bin', 'pickle'], default='bin')
    ap.add_argument('--train-offsets', default='official-data/tinystories/train_bpe_8192_offsets.npy')
    ap.add_argument('--bin-dtype', default='uint16', help='Explicit binary token dtype; inspect official release metadata if layout changes')
    ap.add_argument('--valid', default='official-data/tinystories/valid_bpe_8192.pickle')
    args = ap.parse_args()
    if args.download:
        download(args.download_root, args.models_revision, args.data_revision)
    cfg = load_config(args.config, args.set)
    tk = text_tokenizer(cfg)
    assets = [Path(cfg.paths.artifacts) / cfg.upstream.checkpoints[k] for k in ['bpe_vocab', 'bpe_merges']]
    result = prepare(args.train, args.valid, cfg.paths.data, tk, cfg.ar.context_length, cfg.seed,
                     cfg.ar.vocab_size, asset_paths=assets,
                     train_offsets=args.train_offsets if args.train_format == 'bin' else None,
                     bin_dtype=args.bin_dtype)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
