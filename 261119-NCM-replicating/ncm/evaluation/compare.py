"""Validate comparison conditions, export CSV/JSON and a quality/cost plot."""
import argparse
import csv
import json
from pathlib import Path

from utils.experiment_io import json_write, sha256


def load_run(path):
    root = Path(path)
    generation = json.loads((root / 'generation.json').read_text())
    quality = None
    if (root / 'quality.json').exists():
        quality = json.loads((root / 'quality.json').read_text())
        if quality['generation_sha256'] != sha256(root / 'generation.json'):
            raise ValueError('Quality belongs to a different generation run')
    return generation, quality


def compare(ar, ncm, ar_quality=None, ncm_quality=None, allow_unmatched=False):
    if ar['model'] != 'ar' or ncm['model'] != 'ncm':
        raise ValueError('Expected --ar AR results and --ncm NCM results')
    mismatches = []
    for key in ['length', 'batch_size', 'seeds', 'samples_per_seed', 'precision', 'sampling',
                'cut_at_eos_for_quality', 'fixed_length_for_timing']:
        if ar['protocol'][key] != ncm['protocol'][key]:
            mismatches.append(f'protocol.{key}')
    if ar['assets_sha256'] != ncm['assets_sha256']:
        mismatches.append('text tokenizer')
    for key in ['gpu', 'torch', 'cuda']:
        if ar['environment'][key] != ncm['environment'][key]:
            mismatches.append(f'hardware/software.{key}')
    for key in ['warmup_batches', 'repeats']:
        if ar['benchmark'][key] != ncm['benchmark'][key]:
            mismatches.append(f'benchmark.{key}')
    if (ar_quality is None) != (ncm_quality is None):
        mismatches.append('quality missing for one model')
    if ar_quality is not None and ncm_quality is not None:
        for key in ['ref_model', 'resolved_revision', 'scoring_max_length', 'aggregation']:
            if ar_quality[key] != ncm_quality[key]:
                mismatches.append(f'quality.{key}')
    at, nt = ar['training_provenance'], ncm['training_provenance']
    if at is None or nt is None:
        mismatches.append('training provenance unavailable (e.g. upstream released checkpoint)')
    else:
        for key in ['source_sha256', 'split_sha256', 'seed']:
            if at['data_manifest'][key] != nt['data_manifest'][key]:
                mismatches.append(f'training data.{key}')
        if ar['training_text_positions'] != ncm['training_text_positions']:
            mismatches.append('generator training text positions')
        for key in ['precision', 'seed']:
            if at['config'][key] != nt['config'][key]:
                mismatches.append(f'training.{key}')
        for key in ['generator', 'betas', 'grad_clip']:
            if at['config']['training'][key] != nt['config']['training'][key]:
                mismatches.append(f'training.{key}')
    for gen in [ar, ncm]:
        selected = gen.get('checkpoint_selection', {}).get('step')
        training = gen.get('training_provenance')
        target = training['config']['training']['generator']['steps'] if training else None
        if selected is not None and target is not None and selected < target:
            mismatches.append(f"{gen['model']}: selected checkpoint trained {selected}/{target} updates")
    if mismatches and not allow_unmatched:
        raise ValueError('Comparison conditions differ: ' + ', '.join(mismatches) +
                         '. Use --allow-unmatched only for an explicitly labelled exploratory comparison.')
    rows = []
    for gen, quality in [(ar, ar_quality), (ncm, ncm_quality)]:
        bench = gen['benchmark']
        calls = bench['forward_passes_per_batch']
        rows.append({'model': gen['model'], 'generator_parameters': gen['generator']['parameters'],
                     'tokenizer_parameters': (gen['tokenizer'] or {}).get('parameters', 0),
                     'training_text_positions': gen['training_text_positions'],
                     'selected_checkpoint_step': gen.get('checkpoint_selection', {}).get('step'),
                     'tokenizer_training_text_positions': gen.get('tokenizer_training_text_positions'),
                     'generative_perplexity': quality['generative_perplexity'] if quality else None,
                     'mean_unigram_entropy': quality['mean_unigram_entropy'] if quality else None,
                     'mean_generated_bpe_tokens': quality.get('mean_generated_bpe_tokens') if quality else None,
                     'median_batch_latency_seconds': bench['median_batch_latency_seconds'],
                     'tokens_per_second': bench['tokens_per_second'],
                     'generator_forward_passes_per_batch': sum(x['generator'] for x in calls) / len(calls),
                     'decoder_forward_passes_per_batch': sum(x['decoder'] for x in calls) / len(calls),
                     'peak_allocated_bytes': bench['peak_allocated_bytes'],
                     'peak_reserved_bytes': bench['peak_reserved_bytes']})
    return {'matched_conditions': not mismatches, 'mismatches': mismatches, 'rows': rows,
            'ncm_speedup': rows[0]['median_batch_latency_seconds'] / rows[1]['median_batch_latency_seconds'],
            'budget_scope': 'generator text-position exposure; NCM tokenizer pretraining is additional and must be reported separately',
            'quality_scored': ar_quality is not None and ncm_quality is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ar', required=True)
    ap.add_argument('--ncm', required=True)
    ap.add_argument('--out', default='outputs/comparison')
    ap.add_argument('--allow-unmatched', action='store_true')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()
    ar, aq = load_run(args.ar)
    ncm, nq = load_run(args.ncm)
    result = compare(ar, ncm, aq, nq, args.allow_unmatched)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    json_write(result, root / 'comparison.json')
    with open(root / 'comparison.csv', 'w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['rows'][0]))
        writer.writeheader()
        writer.writerows(result['rows'])
    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        metrics = ['median_batch_latency_seconds', 'tokens_per_second']
        if result['quality_scored']:
            metrics += ['generative_perplexity', 'mean_unigram_entropy']
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            values = [r[metric] for r in result['rows']]
            if any(x is None for x in values):
                ax.text(0.5, 0.5, 'No nonempty samples', ha='center', transform=ax.transAxes)
            else:
                ax.bar(['AR', 'NCM'], values)
            ax.set_title(metric.replace('_', ' '), fontsize=9)
        fig.suptitle('Matched comparison' if result['matched_conditions'] else 'Exploratory: unmatched conditions')
        fig.tight_layout()
        fig.savefig(root / 'comparison.png', dpi=160)
        plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
