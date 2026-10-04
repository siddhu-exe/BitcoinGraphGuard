"""
Lag-safe monitoring channels for BitcoinGraphGuard.

Two channels live here, and both exist because of one rule: **a label that has not
arrived must never influence a decision.** With a label delay of ``L`` steps, the labels
of step ``s`` are usable only from step ``s + L`` onward, so a decision at step ``t`` may
read labels of steps ``<= t - L`` and nothing newer.

1. :class:`LagSafePerformanceMonitor` (label-dependent). Per-step F1 of the deployed
   model on already-labelled steps. Every step counts once (step-weighted, not
   row-weighted), so a large step cannot dilute a crash visible at the next one.
2. :class:`ScoreShiftMonitor` (label-free in the current step). PSI of the deployed
   model's step-``t`` score distribution against its own scores on already-labelled
   steps. It reads no label of step ``t``, so it can fire while labels are delayed.

F1 floor rule (not tuned on the drift window)
---------------------------------------------
``f1_floor = perf_floor_fraction * validation_f1`` with a fixed, pre-declared fraction
(default 0.5). ``validation_f1`` comes from the validation period (steps 25-34) of the
deployed operating point. See :func:`derive_f1_floor`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Fixed probability bins for the score PSI. Declared once, never fitted to a window.
SCORE_BIN_EDGES: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90)

# Standard PSI bands (industry convention, not tuned on this dataset).
PSI_WARNING = 0.10
PSI_CRITICAL = 0.25


class LagViolationError(ValueError):
    """A label-based input contains labels newer than ``time_step - label_delay_steps``."""


def derive_f1_floor(validation_f1: float, fraction: float = 0.5) -> float:
    """
    Derive the PerformanceCrash F1 floor from the validation period.

    The floor is a fixed fraction of the deployed operating point's validation-period F1:
    ``floor = fraction * validation_f1``. Nothing about steps 43-49 enters the rule.

    Parameters
    ----------
    validation_f1 : float
        F1 of the deployed operating point on the validation window (steps 25-34).
        Ideally the median of the per-step validation F1 distribution; when only a
        pooled validation F1 is stored, pass that and say so wherever it is reported.
    fraction : float, default=0.5
        Share of the validated F1 below which a step counts as a crash.
    """
    if not 0.0 < validation_f1 <= 1.0:
        raise ValueError(f"validation_f1 must be in (0, 1], got {validation_f1}")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    return float(fraction * validation_f1)


def psi_from_proportions(
    reference: np.ndarray, target: np.ndarray, min_prob: float = 1e-4
) -> float:
    """Symmetric PSI of two bin-proportion vectors, floored at ``min_prob``."""
    p = np.clip(np.asarray(reference, dtype=float), min_prob, 1.0)
    q = np.clip(np.asarray(target, dtype=float), min_prob, 1.0)
    p, q = p / p.sum(), q / q.sum()
    return float(max(0.0, np.sum((q - p) * np.log(q / p))))


def score_proportions(
    scores: np.ndarray, edges: Sequence[float] = SCORE_BIN_EDGES
) -> np.ndarray:
    """Share of ``scores`` falling in each fixed probability bin."""
    binned = np.digitize(np.asarray(scores, dtype=float), edges)
    counts = np.bincount(binned, minlength=len(edges) + 1)
    return counts / max(1, counts.sum())


class LabelledScoreHistory:
    """
    Per-step (label, score) record of the deployed model, readable only through the lag rule.

    A step may be ``record``-ed as soon as it is scored, but :meth:`eligible_steps` only
    returns steps ``<= time_step - label_delay_steps``. The channels read the history
    exclusively through that method, so recording step ``t`` before evaluating step ``t``
    is harmless: its labels are invisible until step ``t + L``.
    """

    def __init__(self, label_delay_steps: int = 1):
        if label_delay_steps < 0:
            raise ValueError("label_delay_steps must be >= 0")
        self.label_delay_steps = label_delay_steps
        self._steps: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def record(self, time_step: int, y_true: np.ndarray, y_score: np.ndarray) -> None:
        """Store the labels and scores of ``time_step`` (labels stay hidden until ``t + L``)."""
        y = np.asarray(y_true, dtype=int)
        s = np.asarray(y_score, dtype=float)
        if len(y) != len(s):
            raise ValueError("y_true and y_score must have the same length")
        self._steps[time_step] = (y, s)

    def max_usable_step(self, time_step: int) -> int:
        """Newest step whose labels may be read when deciding at ``time_step``."""
        return time_step - self.label_delay_steps

    def eligible_steps(self, time_step: int) -> list[int]:
        """Recorded steps whose labels have arrived by ``time_step`` (strictly ``<= t - L``)."""
        limit = self.max_usable_step(time_step)
        return sorted(s for s in self._steps if s <= limit)

    def get(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        return self._steps[step]


# ---------------------------------------------------------------------------
# Performance channel (label-dependent, lag-safe)
# ---------------------------------------------------------------------------


@dataclass
class PerformanceWindowReport:
    """Lag-safe performance assessment at ``time_step``."""

    time_step: int
    label_delay_steps: int
    max_label_step_allowed: int
    steps_used: list[int]
    per_step_f1: dict[int, float]
    per_step_positives: dict[int, int]
    f1_floor: float
    available: bool
    breached: bool
    f1_min: float = float("nan")
    f1_mean: float = float("nan")
    note: str = ""

    @property
    def max_label_step_used(self) -> int | None:
        return max(self.steps_used) if self.steps_used else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_step": self.time_step,
            "label_delay_steps": self.label_delay_steps,
            "max_label_step_allowed": self.max_label_step_allowed,
            "max_label_step_used": self.max_label_step_used,
            "steps_used": self.steps_used,
            "per_step_f1": {str(k): round(v, 4) for k, v in self.per_step_f1.items()},
            "per_step_positives": {
                str(k): v for k, v in self.per_step_positives.items()
            },
            "f1_min": None if np.isnan(self.f1_min) else round(self.f1_min, 4),
            "f1_mean": None if np.isnan(self.f1_mean) else round(self.f1_mean, 4),
            "f1_floor": round(self.f1_floor, 4),
            "available": self.available,
            "breached": self.breached,
            "note": self.note,
        }


class LagSafePerformanceMonitor:
    """
    PerformanceCrash channel that reads only labels of steps ``<= t - L``.

    The window is the last ``window_steps`` labelled steps holding at least
    ``min_positives`` illicit labels (F1 on fewer positives is single-example noise).
    Each step's F1 is computed separately and counts once; the channel is breached when
    the **worst** of those steps is below ``f1_floor``. A pooled F1 would let a large
    healthy step (e.g. 239 positives) bury a crash at the next one (24 positives).

    Parameters
    ----------
    history : LabelledScoreHistory
        Shared record of labelled steps; also fixes ``label_delay_steps``.
    threshold : float, default=0.435
        Frozen decision threshold of the deployed model.
    f1_floor : float
        Per-step F1 below which a step counts as a crash. Derive it with
        :func:`derive_f1_floor`; do not set it from the drift window.
    window_steps : int, default=2
        Number of most recent eligible steps examined.
    min_positives : int, default=10
        Minimum illicit labels for a step to be eligible.
    """

    def __init__(
        self,
        history: LabelledScoreHistory,
        f1_floor: float,
        threshold: float = 0.435,
        window_steps: int = 2,
        min_positives: int = 10,
    ):
        if window_steps < 1:
            raise ValueError("window_steps must be >= 1")
        self.history = history
        self.f1_floor = f1_floor
        self.threshold = threshold
        self.window_steps = window_steps
        self.min_positives = min_positives

    @staticmethod
    def _f1(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
        pred = y_score >= threshold
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true != 1)))
        fn = int(np.sum(~pred & (y_true == 1)))
        denom = 2 * tp + fp + fn
        return float(2 * tp / denom) if denom > 0 else 0.0

    def evaluate(self, time_step: int) -> PerformanceWindowReport:
        """Assess the deployed model at ``time_step`` from labels of steps ``<= t - L`` only."""
        allowed = self.history.max_usable_step(time_step)
        eligible = [
            s
            for s in self.history.eligible_steps(time_step)
            if int(np.sum(self.history.get(s)[0] == 1)) >= self.min_positives
        ]
        used = eligible[-self.window_steps :]
        if not used:
            return PerformanceWindowReport(
                time_step=time_step,
                label_delay_steps=self.history.label_delay_steps,
                max_label_step_allowed=allowed,
                steps_used=[],
                per_step_f1={},
                per_step_positives={},
                f1_floor=self.f1_floor,
                available=False,
                breached=False,
                note=f"No labelled step <= {allowed} with >= {self.min_positives} illicit labels yet.",
            )

        per_step_f1: dict[int, float] = {}
        per_step_pos: dict[int, int] = {}
        for s in used:
            y, sc = self.history.get(s)
            per_step_f1[s] = self._f1(y, sc, self.threshold)
            per_step_pos[s] = int(np.sum(y == 1))
        f1s = list(per_step_f1.values())
        f1_min, f1_mean = float(min(f1s)), float(np.mean(f1s))
        return PerformanceWindowReport(
            time_step=time_step,
            label_delay_steps=self.history.label_delay_steps,
            max_label_step_allowed=allowed,
            steps_used=used,
            per_step_f1=per_step_f1,
            per_step_positives=per_step_pos,
            f1_floor=self.f1_floor,
            available=True,
            breached=f1_min < self.f1_floor,
            f1_min=f1_min,
            f1_mean=f1_mean,
        )


# ---------------------------------------------------------------------------
# Score-shift channel (label-free)
# ---------------------------------------------------------------------------


@dataclass
class ScoreShiftReport:
    """PSI of the deployed model's step-``t`` scores against its scores on labelled steps."""

    time_step: int
    available: bool
    psi: float = float("nan")
    reference_steps: list[int] = field(default_factory=list)
    n_current_scores: int = 0
    warning_threshold: float = PSI_WARNING
    critical_threshold: float = PSI_CRITICAL
    note: str = ""

    @property
    def level(self) -> str:
        if not self.available:
            return "UNAVAILABLE"
        if self.psi >= self.critical_threshold:
            return "CRITICAL"
        if self.psi >= self.warning_threshold:
            return "WARNING"
        return "STABLE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_step": self.time_step,
            "available": self.available,
            "psi": None if np.isnan(self.psi) else round(self.psi, 4),
            "level": self.level,
            "reference_steps": self.reference_steps,
            "n_current_scores": self.n_current_scores,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "note": self.note,
        }


class ScoreShiftMonitor:
    """
    Label-free score-distribution shift channel.

    Reference = the deployed model's scores on the last ``reference_steps`` steps whose
    labels have already arrived (steps ``<= t - L``), each step weighted equally. The
    reference is therefore a period in which the model's behaviour was verifiable. The
    current step contributes scores only; its labels are never read, so the channel works
    during label delay. Bands are the standard PSI 0.10 / 0.25, not tuned on a window.

    Parameters
    ----------
    history : LabelledScoreHistory
        Shared record of labelled steps; fixes ``label_delay_steps``.
    reference_steps : int, default=5
        How many of the most recent labelled steps form the reference.
    min_reference_steps : int, default=2
        Fewer labelled steps than this and the channel reports UNAVAILABLE (never fires).
    """

    def __init__(
        self,
        history: LabelledScoreHistory,
        reference_steps: int = 5,
        min_reference_steps: int = 2,
        warning_threshold: float = PSI_WARNING,
        critical_threshold: float = PSI_CRITICAL,
    ):
        if min_reference_steps < 1 or reference_steps < min_reference_steps:
            raise ValueError("need 1 <= min_reference_steps <= reference_steps")
        self.history = history
        self.reference_steps = reference_steps
        self.min_reference_steps = min_reference_steps
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def evaluate(self, time_step: int, current_scores: np.ndarray) -> ScoreShiftReport:
        """PSI of ``current_scores`` (step ``t``, no labels) against the labelled-step reference."""
        ref = self.history.eligible_steps(time_step)[-self.reference_steps :]
        n_cur = len(current_scores)
        if len(ref) < self.min_reference_steps or n_cur == 0:
            return ScoreShiftReport(
                time_step=time_step,
                available=False,
                reference_steps=ref,
                n_current_scores=n_cur,
                warning_threshold=self.warning_threshold,
                critical_threshold=self.critical_threshold,
                note=(
                    f"Need >= {self.min_reference_steps} labelled reference steps "
                    f"(<= {self.history.max_usable_step(time_step)}); have {len(ref)}."
                ),
            )
        ref_props = np.mean(
            [score_proportions(self.history.get(s)[1]) for s in ref], axis=0
        )
        psi = psi_from_proportions(ref_props, score_proportions(current_scores))
        return ScoreShiftReport(
            time_step=time_step,
            available=True,
            psi=psi,
            reference_steps=ref,
            n_current_scores=n_cur,
            warning_threshold=self.warning_threshold,
            critical_threshold=self.critical_threshold,
        )
