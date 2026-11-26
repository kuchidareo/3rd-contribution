"""Offline tests for notebook profile selection; no torch or dataset required."""
from copy import deepcopy
import unittest

from training.profiles import PAPER_GENERATOR, configure_profile, check_checkpoint_progress


def inputs():
    cfg = {'seed': 7, 'precision': 'fp32', 'paths': {'data': '/kaggle/input/existing/prepared'},
           'ar': {}, 'training': {'generator': {}, 'tokenizer': {'steps': 30000}},
           'sampling': {}, 'evaluation': {}}
    official = {'model': dict(n_layer=12, n_head=12, n_embd=768, bias=True, dropout=.1,
                              use_qk_norm=True, rope_base=100000),
                'tokenizer': {'vqvae': {'context_length': 128}},
                'generation': {'temperature': 1., 'top_k': 0, 'top_p': 1.},
                'eval': {'seeds': [42, 0, 1, 2, 3], 'n_samples': 256,
                         'batch_size': 128, 'ref_model': 'gpt2-large'}}
    return cfg, official


class ProfileTests(unittest.TestCase):
    def test_small_profile_matches_generators_and_preserves_tokenizer(self):
        cfg, official = inputs()
        result = configure_profile(cfg, official, 'small', bf16_supported=False,
                                   precision='fp16', micro_batch=2)
        for key, value in dict(n_layer=6, n_embd=384, n_head=6).items():
            self.assertEqual(result['ar'][key], value)
            self.assertEqual(result['upstream_overrides']['model'][key], value)
        self.assertNotIn('tokenizer', result['upstream_overrides'])
        self.assertEqual(result['training']['tokenizer'], cfg['training']['tokenizer'])
        self.assertEqual(result['training']['generator']['warmup_steps'], 100)
        self.assertEqual(result['training']['generator']['global_batch_size'], 64)
        self.assertEqual(result['experiment']['gradient_accumulation_steps'], 32)
        self.assertTrue(result['ncm_selective_fp32'])

    def test_paper_published_settings_and_budget(self):
        cfg, official = inputs()
        original = deepcopy(cfg)
        result = configure_profile(cfg, official, 'paper', bf16_supported=True)
        for key, value in PAPER_GENERATOR.items():
            self.assertEqual(result['training']['generator'][key], value)
        self.assertEqual(result['precision'], 'bf16')
        self.assertFalse(result['ncm_selective_fp32'])
        self.assertTrue(result['training']['compile'])
        self.assertEqual(result['training']['betas'], [.9, .95])
        self.assertEqual(result['experiment']['gradient_accumulation_steps'], 1024)
        self.assertEqual(result['experiment']['target_generator_text_positions'], 13107200000)
        self.assertEqual(result['evaluation']['batch_size'], 128)
        self.assertEqual(result['evaluation']['samples_per_seed'], 256)
        self.assertEqual(result['evaluation']['seeds'], [42, 0, 1, 2, 3])
        self.assertEqual(result['evaluation']['scoring_batch_size'], 32)
        self.assertEqual(result['paths'], original['paths'])
        self.assertEqual(cfg, original)
        self.assertFalse(result['experiment']['full_paper_equivalence_verified'])
        self.assertIn('padding_loss', result['experiment']['training_assumptions'])

    def test_paper_does_not_silently_fall_back(self):
        cfg, official = inputs()
        with self.assertRaisesRegex(RuntimeError, 'BF16-capable'):
            configure_profile(cfg, official, 'paper', bf16_supported=False)

    def test_t4_fp16_keeps_full_budget_and_records_deviation(self):
        cfg, official = inputs()
        result = configure_profile(cfg, official, 'paper', bf16_supported=False, precision='fp16')
        self.assertEqual(result['precision'], 'fp16')
        self.assertTrue(result['ncm_selective_fp32'])
        self.assertIn('FP32', result['experiment']['ncm_precision_policy'])
        self.assertEqual(result['training']['generator']['steps'], 50000)
        self.assertEqual(result['training']['generator']['global_batch_size'], 2048)
        self.assertEqual(result['evaluation']['batch_size'], 128)
        self.assertEqual(result['experiment']['target_generator_text_positions'], 13107200000)
        self.assertFalse(result['experiment']['paper_precision_matched'])
        self.assertEqual(len(result['experiment']['hardware_deviations']), 1)
        ck = {'kind': 'ncm', 'step': 1000, 'metadata': {'config': result}}
        self.assertEqual(check_checkpoint_progress('ncm', ck, 'paper', 50000, precision='fp16'), 1000)
        with self.assertRaisesRegex(ValueError, 'same precision'):
            check_checkpoint_progress('ncm', ck, 'paper', 50000, precision='bf16')

    def test_paper_rejects_unrecognized_precision(self):
        cfg, official = inputs()
        for precision in ['fp32', 'int8']:
            with self.assertRaises(ValueError):
                configure_profile(cfg, official, 'paper', bf16_supported=True, precision=precision)

    def test_paper_rejects_architecture_override(self):
        cfg, official = inputs()
        with self.assertRaisesRegex(ValueError, 'unmodified'):
            configure_profile(cfg, official, 'paper', bf16_supported=True, model_overrides={'n_layer': 1})
        cfg['upstream_overrides'] = {'tokenizer': {'vqvae': {'embed_dim': 16}}}
        with self.assertRaises(ValueError):
            configure_profile(cfg, official, 'paper', bf16_supported=True)

    def test_pilot_is_small_and_hardware_flexible(self):
        cfg, official = inputs()
        result = configure_profile(cfg, official, 'pilot', bf16_supported=False)
        self.assertEqual(result['precision'], 'fp32')
        self.assertEqual(result['training']['generator']['steps'], 100)
        self.assertEqual(result['training']['generator']['global_batch_size'], 32)
        self.assertEqual(result['experiment']['target_generator_text_positions'], 409600)
        self.assertFalse(result['training']['compile'])

    def test_invalid_profile_and_accumulation(self):
        cfg, official = inputs()
        with self.assertRaises(ValueError):
            configure_profile(cfg, official, 'typo', bf16_supported=True)
        with self.assertRaisesRegex(ValueError, 'divide'):
            configure_profile(cfg, official, 'paper', bf16_supported=True, micro_batch=3)

    def test_matching_paper_checkpoint_and_incomplete_progress(self):
        cfg, official = inputs()
        config = configure_profile(cfg, official, 'paper', bf16_supported=True)
        ck = {'kind': 'ncm', 'step': 1000, 'metadata': {'config': config}}
        self.assertEqual(check_checkpoint_progress('ncm', ck, 'paper', 50000), 1000)
        ck['step'] = 50000
        self.assertEqual(check_checkpoint_progress('ncm', ck, 'paper', 50000), 50000)
        ck['step'] = 50001
        with self.assertRaises(ValueError):
            check_checkpoint_progress('ncm', ck, 'paper', 50000)

    def test_pilot_checkpoint_cannot_be_relabelled_paper(self):
        cfg, official = inputs()
        config = configure_profile(cfg, official, 'pilot', bf16_supported=True)
        ck = {'kind': 'ar', 'step': 100, 'metadata': {'config': config}}
        with self.assertRaisesRegex(ValueError, 'pilot'):
            check_checkpoint_progress('ar', ck, 'paper', 50000)
        with self.assertRaisesRegex(ValueError, 'kind'):
            check_checkpoint_progress('ncm', ck, 'pilot', 100)


class NotebookWiringTests(unittest.TestCase):
    @staticmethod
    def cells():
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / 'notebooks/02_train_tinystories_kaggle.ipynb'
        return json.loads(path.read_text())['cells']

    def test_configuration_calls_tested_profile_helper(self):
        source = next(c['source'] for c in self.cells() if c['cell_type'] == 'code' and 'PROFILE =' in c['source'])
        self.assertIn('OmegaConf.create(configure_profile(', source)
        self.assertIn("PROFILE = 'small'", source)
        self.assertIn("STOP_AT_UPDATE = 100", source)
        self.assertIn('precision=PRECISION', source)
        self.assertIn('torch.cuda.get_device_capability()[0] >= 8', source)
        self.assertNotIn("cfg.precision = 'bf16' if", source)

    def test_notebook_refreshes_a_stale_imported_function(self):
        import sys
        from pathlib import Path
        import training.profiles as module
        source = next(c['source'] for c in self.cells() if c['cell_type'] == 'code' and 'PROFILE =' in c['source'])
        start = source.index('# Reload after Git checkout:')
        end = source.index('cfg = OmegaConf.load', start)
        old = module.configure_profile
        try:
            module.configure_profile = lambda config, official, profile: None
            context = {'Path': Path, 'REPO': Path(__file__).resolve().parents[1]}
            exec(compile(source[start:end], '<notebook-profile-reload>', 'exec'), context)
            import inspect
            self.assertIn('precision', inspect.signature(context['configure_profile']).parameters)
            self.assertIs(context['configure_profile'], module.configure_profile)
        finally:
            module.configure_profile = old

    def run_evaluation_cell(self, ar_step, ncm_step, early_stopped=False):
        from pathlib import Path
        import sys
        from types import SimpleNamespace
        import json
        archived = (Path(__file__).resolve().parents[1] /
                    'notebooks/big-model-backup/02_train_tinystories_kaggle.ipynb')
        cells = json.loads(archived.read_text())['cells']
        cell = next(c for c in cells if c['cell_type'] == 'code' and 'RESULT_TAG =' in c['source'])
        cfg, official = inputs()
        config = configure_profile(cfg, official, 'paper', bf16_supported=True)
        def load(path, **kwargs):
            kind = path.parent.name
            return {'kind': kind, 'step': ar_step if kind == 'ar' else ncm_step,
                    'stop_reason': 'early_stopping' if early_stopped else 'session_limit',
                    'metadata': {'config': config}}
        commands = []
        context = dict(PROFILE='paper', WORK=Path('/example/paper'), CONFIG=Path('/example/paper/experiment.yaml'),
                       cfg=SimpleNamespace(precision='bf16', training=SimpleNamespace(generator=SimpleNamespace(steps=50000),
                               early_stopping=SimpleNamespace(enabled=early_stopped))),
                       EVALUATION_CHECKPOINT='best',
                       torch=SimpleNamespace(load=load), check_checkpoint_progress=check_checkpoint_progress,
                       sys=sys, run=lambda command: commands.append(command))
        exec(compile(cell['source'], '<notebook-final-evaluation>', 'exec'), context)
        return commands

    def test_partial_paper_training_cannot_run_final_evaluation(self):
        with self.assertRaisesRegex(RuntimeError, '1000/50000'):
            self.run_evaluation_cell(1000, 50000)
        with self.assertRaisesRegex(RuntimeError, '1000/50000'):
            self.run_evaluation_cell(50000, 1000)

    def test_early_stopped_models_can_evaluate_best_checkpoints(self):
        commands = self.run_evaluation_cell(700, 900, early_stopped=True)
        self.assertEqual(len(commands), 2)
        self.assertTrue(all(str(command[command.index('--checkpoint') + 1]).endswith('best.pt') for command in commands))

    def test_finished_paper_runs_launch_both_evaluations(self):
        commands = self.run_evaluation_cell(50000, 50000)
        self.assertEqual(len(commands), 2)
        self.assertIn('evaluation.evaluate', commands[0])
        self.assertIn('ar', commands[0])
        self.assertIn('ncm', commands[1])


if __name__ == '__main__':
    unittest.main()
