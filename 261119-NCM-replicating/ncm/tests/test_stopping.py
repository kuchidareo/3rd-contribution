"""Validation selection, patience and old-checkpoint compatibility without torch."""
from copy import deepcopy
import unittest

from training.stopping import EarlyStopping, compatible_resume
from utils.experiment_io import fingerprint


class StoppingTests(unittest.TestCase):
    def test_lowest_loss_is_selected_even_for_small_improvement(self):
        stop = EarlyStopping(patience=2, min_delta=.01, min_steps=0)
        self.assertTrue(stop.observe(2., 100))
        self.assertTrue(stop.observe(1.995, 200))
        self.assertEqual(stop.best_step, 200)
        self.assertEqual(stop.bad_checks, 1)
        self.assertFalse(stop.observe(2.1, 300))
        self.assertTrue(stop.stopped)
        self.assertEqual(stop.best_loss, 1.995)

    def test_significant_improvement_resets_patience(self):
        stop = EarlyStopping(patience=2, min_delta=.01, min_steps=0)
        for loss, step in [(2., 100), (2.1, 200), (1.8, 300)]:
            stop.observe(loss, step)
        self.assertEqual(stop.bad_checks, 0)
        self.assertFalse(stop.stopped)

    def test_no_stopping_before_minimum(self):
        stop = EarlyStopping(patience=2, min_steps=500)
        for step in [100, 200, 300, 400]:
            stop.observe(2., step)
            self.assertFalse(stop.stopped)
        stop.observe(2., 500)
        self.assertFalse(stop.stopped)
        stop.observe(2., 600)
        self.assertTrue(stop.stopped)

    def test_disabling_stopping_preserves_best_selection(self):
        stop = EarlyStopping(enabled=False, patience=1, min_steps=0)
        stop.observe(1., 100)
        stop.observe(2., 200)
        self.assertFalse(stop.stopped)
        self.assertEqual(stop.best_step, 100)

    def test_resume_keeps_patience_best_and_stop_state(self):
        stop = EarlyStopping(patience=2, min_steps=0)
        stop.observe(1., 100)
        stop.observe(2., 200)
        resumed = EarlyStopping.restore(stop.state_dict(), **stop.policy)
        self.assertFalse(resumed.observe(3., 300))
        self.assertTrue(resumed.stopped)
        self.assertEqual(resumed.best_step, 100)
        stopped = EarlyStopping.restore(resumed.state_dict(), **stop.policy)
        self.assertTrue(stopped.stopped)

    def test_explicit_policy_change_restarts_patience_not_best(self):
        stop = EarlyStopping(patience=1, min_steps=0)
        stop.observe(1., 100)
        stop.observe(2., 200)
        resumed = EarlyStopping.restore(stop.state_dict(), enabled=True, patience=5, min_delta=.001, min_steps=0)
        self.assertFalse(resumed.stopped)
        self.assertEqual(resumed.bad_checks, 0)
        self.assertEqual(resumed.best_loss, 1.)

    def test_invalid_validation_and_policy(self):
        for value in [float('inf'), float('nan')]:
            with self.assertRaises(FloatingPointError):
                EarlyStopping().observe(value, 100)
        with self.assertRaises(ValueError):
            EarlyStopping(patience=0)

    def test_old_checkpoint_accepts_new_monitoring_but_not_new_optimizer(self):
        old = dict(kind='ar', recipe={'steps': 50000, 'lr': .0003},
                   training={'generator': {'steps': 50000, 'lr': .0003},
                             'validation_every': 500, 'checkpoint_every': 500,
                             'validation_rows': 256}, data_manifest_sha256='same')
        ck = {'signature_hash': fingerprint(old), 'metadata': {'config': {'training': deepcopy(old['training'])}}}
        current = deepcopy(old)
        current['training'].update(validation_every=100, checkpoint_every=100,
                                   early_stopping=dict(enabled=True, patience=5))
        self.assertTrue(compatible_resume(current, ck))
        current['recipe']['lr'] = .001
        self.assertFalse(compatible_resume(current, ck))
        current = deepcopy(old)
        current['training']['validation_rows'] = 64
        self.assertFalse(compatible_resume(current, ck))


if __name__ == '__main__':
    unittest.main()
