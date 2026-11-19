"""Dependency-free comparison guards; runnable with the workspace venv."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from evaluation.compare import compare
from utils.experiment_io import json_write


def fixture(kind):
    training = {'data_manifest': {'source_sha256': {'source': 'abc'},
                                'split_sha256': {'train': 'def'}, 'seed': 42},
                'config': {'seed': 42, 'precision': 'fp32',
                           'training': {'generator': {'steps': 2}, 'betas': [.9, .95], 'grad_clip': 1}}}
    return {'model': kind, 'protocol': dict(length=128, batch_size=16, seeds=[42], samples_per_seed=2,
            precision='fp32', sampling={}, cut_at_eos_for_quality=True, fixed_length_for_timing=True),
            'assets_sha256': {'vocab': 'abc'}, 'environment': dict(gpu=None, torch='test', cuda=None),
            'benchmark': dict(warmup_batches=1, repeats=2,
                              forward_passes_per_batch=[dict(generator=128 if kind == 'ar' else 11,
                                                             decoder=0 if kind == 'ar' else 1)],
                              median_batch_latency_seconds=2. if kind == 'ar' else 1.,
                              tokens_per_second=1024. if kind == 'ar' else 2048.,
                              peak_allocated_bytes=None, peak_reserved_bytes=None),
            'training_provenance': training, 'training_text_positions': 2048,
            'generator': {'parameters': 1}, 'tokenizer': None}


class ComparisonTests(unittest.TestCase):
    def test_matching_and_budget_guard(self):
        a, b = fixture('ar'), fixture('ncm')
        result = compare(a, b)
        self.assertTrue(result['matched_conditions'])
        self.assertEqual(result['ncm_speedup'], 2)
        b['training_text_positions'] += 1
        with self.assertRaisesRegex(ValueError, 'training text positions'):
            compare(a, b)

    def test_protocol_hardware_and_unknown_provenance(self):
        a, b = fixture('ar'), fixture('ncm')
        b['protocol']['batch_size'] = 8
        b['environment']['gpu'] = 'different GPU'
        b['training_provenance'] = None
        with self.assertRaisesRegex(ValueError, 'protocol.batch_size'):
            compare(a, b)
        result = compare(a, b, allow_unmatched=True)
        self.assertFalse(result['matched_conditions'])
        self.assertEqual(len(result['mismatches']), 3)

    def test_reference_revision_guard(self):
        q = dict(ref_model='gpt2-large', resolved_revision='abc', scoring_max_length=1024,
                 aggregation='same', generative_perplexity=20., mean_unigram_entropy=4.)
        other = copy.deepcopy(q)
        other['resolved_revision'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'resolved_revision'):
            compare(fixture('ar'), fixture('ncm'), q, other)

    def test_partial_best_checkpoints_are_labelled(self):
        a, b = fixture('ar'), fixture('ncm')
        a['checkpoint_selection'] = {'step': 1}
        b['checkpoint_selection'] = {'step': 1}
        with self.assertRaisesRegex(ValueError, '1/2 updates'):
            compare(a, b)
        result = compare(a, b, allow_unmatched=True)
        self.assertFalse(result['matched_conditions'])
        self.assertEqual(result['rows'][0]['selected_checkpoint_step'], 1)

    def test_cli_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for kind in ['ar', 'ncm']:
                json_write(fixture(kind), root / kind / 'generation.json')
            command = [sys.executable, '-m', 'evaluation.compare', '--ar', str(root / 'ar'),
                       '--ncm', str(root / 'ncm'), '--out', str(root / 'comparison'), '--no-plot']
            subprocess.run(command, check=True, capture_output=True)
            result = json.loads((root / 'comparison/comparison.json').read_text())
            self.assertTrue(result['matched_conditions'])
            self.assertIn('tokens_per_second', (root / 'comparison/comparison.csv').read_text())


if __name__ == '__main__':
    unittest.main()
