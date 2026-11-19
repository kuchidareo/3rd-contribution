"""Validation selection and resume-compatible operational controls; no torch dependency."""
from copy import deepcopy
import math

from utils.experiment_io import fingerprint


class EarlyStopping:
    def __init__(self, *, enabled=True, patience=5, min_delta=0.001, min_steps=500):
        if patience < 1 or not math.isfinite(min_delta) or min_delta < 0 or min_steps < 0:
            raise ValueError('Require patience >= 1, min_delta >= 0, min_steps >= 0')
        self.policy = dict(enabled=bool(enabled), patience=int(patience),
                           min_delta=float(min_delta), min_steps=int(min_steps))
        self.best_loss = None
        self.best_step = None
        self.reference_loss = None
        self.bad_checks = 0
        self.stopped = False

    def observe(self, loss, step):
        loss = float(loss)
        if not math.isfinite(loss):
            raise FloatingPointError('Nonfinite validation loss; refusing checkpoint selection')
        improved = self.best_loss is None or loss < self.best_loss
        if improved:
            self.best_loss, self.best_step = loss, int(step)
        significant = self.reference_loss is None or loss < self.reference_loss - self.policy['min_delta']
        if significant:
            self.reference_loss = loss
            self.bad_checks = 0
        elif step >= self.policy['min_steps']:
            self.bad_checks += 1
        else:
            self.bad_checks = 0
        self.stopped = bool(self.policy['enabled'] and step >= self.policy['min_steps'] and
                            self.bad_checks >= self.policy['patience'])
        return improved

    def state_dict(self):
        return dict(policy=dict(self.policy), best_loss=self.best_loss, best_step=self.best_step,
                    reference_loss=self.reference_loss, bad_checks=self.bad_checks, stopped=self.stopped)

    @classmethod
    def restore(cls, state, **policy):
        result = cls(**policy)
        if state:
            for name in ['best_loss', 'best_step', 'reference_loss', 'bad_checks', 'stopped']:
                setattr(result, name, state[name])
            if result.policy != state['policy']:
                result.bad_checks = 0
                result.stopped = False
                result.reference_loss = result.best_loss
        return result


def compatible_resume(signature, checkpoint):
    """Only monitoring/checkpoint controls may change; optimization/data remain locked."""
    if fingerprint(signature) == checkpoint['signature_hash']:
        return True
    saved_training = checkpoint.get('metadata', {}).get('config', {}).get('training', {})
    candidate = deepcopy(signature)
    for key in ['validation_every', 'checkpoint_every', 'log_every', 'early_stopping']:
        if key in saved_training:
            candidate['training'][key] = deepcopy(saved_training[key])
        else:
            candidate['training'].pop(key, None)
    return fingerprint(candidate) == checkpoint['signature_hash']
