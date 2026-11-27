"""Pure training-schedule arithmetic (no torch), unit-tested."""

from __future__ import annotations

import math


def total_optimizer_steps(n_examples: int, batch_size: int, grad_accum: int, epochs: float) -> int:
    steps_per_epoch = math.ceil(n_examples / (batch_size * grad_accum))
    return max(1, math.ceil(steps_per_epoch * epochs))


def effective_eval_steps(total_steps: int, configured: int, min_evals: int = 4) -> int:
    """Evaluation interval actually used.

    With few examples (data-size ablation) a run can be shorter than the configured interval,
    which would mean no evaluation and no best-checkpoint selection. The interval is then
    shrunk so that every run is evaluated at least ``min_evals`` times.
    """
    return max(1, min(configured, total_steps // min_evals))
