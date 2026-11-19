"""Optimizer-step accounting for dynamic loss scaling (independent of model structure)."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class UpdateResult:
    applied: bool
    grad_norm: float
    scale_before: float
    scale_after: float


def optimizer_update(optimizer, scaler, clip_gradients):
    """Return whether weights changed; callers retry rejected accumulated batches."""
    scaler.unscale_(optimizer)
    norm = float(clip_gradients())
    before = float(scaler.get_scale())
    if not math.isfinite(norm):
        if not scaler.is_enabled():
            raise FloatingPointError('Nonfinite gradient norm without FP16 loss scaling')
        # No optimizer step, even if only the norm reduction overflowed. Explicit
        # update also clears GradScaler's per-optimizer state for the retry.
        scaler.update(new_scale=before * 0.5)
        return UpdateResult(False, norm, before, float(scaler.get_scale()))
    scaler.step(optimizer)
    scaler.update()
    after = float(scaler.get_scale())
    return UpdateResult(after >= before, norm, before, after)
