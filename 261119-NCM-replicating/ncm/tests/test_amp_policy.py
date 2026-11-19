"""Exercise optimizer-step/overflow policy without requiring a GPU or torch."""
import unittest

from training.amp import optimizer_update


class Optimizer:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class Scaler:
    def __init__(self, enabled=True, skip=False, growth=False):
        self.value = 1024.0 if enabled else 1.0
        self.enabled = enabled
        self.skip = skip
        self.growth = growth
        self.unscaled = False

    def is_enabled(self):
        return self.enabled

    def get_scale(self):
        return self.value

    def unscale_(self, optimizer):
        if self.unscaled:
            raise RuntimeError('unscale_ called twice without update')
        self.unscaled = True

    def step(self, optimizer):
        if not self.skip:
            optimizer.step()

    def update(self, new_scale=None):
        if new_scale is not None:
            self.value = new_scale
        elif self.skip:
            self.value *= .5
        elif self.growth:
            self.value *= 2
        self.unscaled = False


class AmpPolicyTests(unittest.TestCase):
    def test_overflow_does_not_change_weights_then_retry_applies(self):
        optimizer, scaler = Optimizer(), Scaler()
        rejected = optimizer_update(optimizer, scaler, lambda: float('inf'))
        self.assertFalse(rejected.applied)
        self.assertEqual(optimizer.steps, 0)
        self.assertEqual(scaler.get_scale(), 512.)
        accepted = optimizer_update(optimizer, scaler, lambda: 1.)
        self.assertTrue(accepted.applied)
        self.assertEqual(optimizer.steps, 1)
        self.assertEqual(accepted.grad_norm, 1.)

    def test_nan_norm_also_rejected(self):
        optimizer, scaler = Optimizer(), Scaler()
        self.assertFalse(optimizer_update(optimizer, scaler, lambda: float('nan')).applied)
        self.assertEqual(optimizer.steps, 0)

    def test_scaler_skip_not_counted_as_an_update(self):
        optimizer, scaler = Optimizer(), Scaler(skip=True)
        self.assertFalse(optimizer_update(optimizer, scaler, lambda: 1.).applied)
        self.assertEqual(optimizer.steps, 0)

    def test_scale_growth_is_a_successful_update(self):
        optimizer, scaler = Optimizer(), Scaler(growth=True)
        self.assertTrue(optimizer_update(optimizer, scaler, lambda: 1.).applied)
        self.assertEqual(optimizer.steps, 1)

    def test_fp32_nonfinite_norm_is_not_retried(self):
        optimizer, scaler = Optimizer(), Scaler(enabled=False)
        with self.assertRaises(FloatingPointError):
            optimizer_update(optimizer, scaler, lambda: float('inf'))
        self.assertEqual(optimizer.steps, 0)


if __name__ == '__main__':
    unittest.main()
