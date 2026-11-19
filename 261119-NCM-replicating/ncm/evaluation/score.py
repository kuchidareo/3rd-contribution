"""Reference scoring in a fresh process after generator GPU memory is released."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import torch

from training.runtime import environment, json_write, sha256


def unigram_entropy(ids):
    if not ids:
        return None
    counts = Counter(ids)
    n = len(ids)
    return -sum((value / n) * math.log(value / n) for value in counts.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--ref-model', default=None, help='Override recorded GPT-2-large scorer (deviation)')
    ap.add_argument('--ref-revision', default='main')
    ap.add_argument('--batch-size', type=int, default=None)
    args = ap.parse_args()
    root = Path(args.run)
    meta = json.loads((root / 'generation.json').read_text())
    if sha256(root / 'samples.jsonl') != meta['samples_sha256']:
        raise ValueError('Samples changed since generation')
    rows = [json.loads(line) for line in (root / 'samples.jsonl').read_text().splitlines()]
    ec = meta['config']['evaluation']
    ref_name = args.ref_model or ec['ref_model']
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluation.gen_ppl import generation_perplexity
    tokenizer = AutoTokenizer.from_pretrained(ref_name, revision=args.ref_revision)
    ref = AutoModelForCausalLM.from_pretrained(ref_name, revision=args.ref_revision).to(args.device).eval()
    by_seed = []
    for seed in meta['protocol']['seeds']:
        selected = [row for row in rows if row['seed'] == seed]
        if len(selected) != meta['protocol']['samples_per_seed']:
            raise ValueError('Sample count does not match generation protocol')
        texts = [row['text'] for row in selected]
        with torch.no_grad():
            ppl = generation_perplexity(ref, tokenizer, [''] * len(texts), texts, device=args.device,
                                        batch_size=args.batch_size or ec['scoring_batch_size'],
                                        max_length=ec['scoring_max_length'])
        if ppl.token_count.sum() == 0:
            raise ValueError('No scorable generated tokens; refusing to report perplexity=1')
        entropy = [unigram_entropy(row['entropy_ids']) for row in selected]
        valid_entropy = [x for x in entropy if x is not None]
        by_seed.append({'seed': seed, 'generative_perplexity': ppl.mean_ppl,
                        'mean_unigram_entropy': float(np.mean(valid_entropy)) if valid_entropy else None,
                        'empty_texts': sum(not t for t in texts),
                        'mean_generated_bpe_tokens': float(np.mean([len(row['entropy_ids']) for row in selected])),
                        'scored_tokens': int(ppl.token_count.sum()),
                        'zero_scored_token_samples': int((ppl.token_count == 0).sum())})
    pp = [row['generative_perplexity'] for row in by_seed]
    ent = [row['mean_unigram_entropy'] for row in by_seed if row['mean_unigram_entropy'] is not None]
    json_write({'ref_model': ref_name, 'requested_revision': args.ref_revision,
                'resolved_revision': getattr(ref.config, '_commit_hash', None),
                'scoring_max_length': ec['scoring_max_length'], 'environment': environment(),
                'generation_sha256': sha256(root / 'generation.json'),
                'by_seed': by_seed, 'generative_perplexity': float(np.mean(pp)),
                'generative_perplexity_std': float(np.std(pp, ddof=1)) if len(pp) > 1 else 0.0,
                'mean_unigram_entropy': float(np.mean(ent)) if ent else None,
                'mean_generated_bpe_tokens': float(np.mean([len(row['entropy_ids']) for row in rows])),
                'aggregation': 'corpus token-weighted PPL per seed; arithmetic mean over seeds; entropy excludes empty sequences'},
               root / 'quality.json')
    print((root / 'quality.json').read_text())


if __name__ == '__main__':
    main()
