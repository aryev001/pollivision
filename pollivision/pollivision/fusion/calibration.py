"""Probability calibration and evidence fusion helpers.

Cue scores arrive on incompatible scales: a CLIP softmax is overconfident, a
geometric morphology score is a bounded heuristic, and a trained probe is
roughly calibrated. Fusing them by averaging probabilities would let the
overconfident cue dominate. Instead every cue is converted to a log-odds
(logit), scaled by a trust weight, and summed - the standard naive-Bayes
combination, which handles disagreement gracefully and keeps each cue's
contribution auditable in ``ScoreBreakdown``.
"""

from __future__ import annotations

import math

import numpy as np

from ..types import ScoreBreakdown

EPS = 1e-6


def to_logit(probability: float, clip: float = 0.995) -> float:
    """Convert a probability to log-odds, clipped to keep the result finite."""
    p = float(np.clip(probability, 1.0 - clip, clip))
    return math.log(p / (1.0 - p))


def to_probability(logit: float) -> float:
    """Inverse of :func:`to_logit`, numerically safe for large magnitudes."""
    if logit >= 0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def temperature_scale(probability: float, temperature: float) -> float:
    """Soften (T>1) or sharpen (T<1) a probability in logit space.

    Zero-shot vision-language softmaxes are systematically overconfident, so the
    VLM cue is passed through this with T>1 before fusion.
    """
    if temperature <= 0:
        return probability
    return to_probability(to_logit(probability) / temperature)


def fuse_logits(evidence: dict[str, tuple[float, float]],
                prior: float = 0.5) -> tuple[float, ScoreBreakdown]:
    """Combine weighted probabilistic cues into one posterior.

    Args:
        evidence: ``{cue_name: (probability, weight)}``. A weight of 0 or a
            probability of exactly 0.5 contributes nothing, so heads can pass a
            neutral value instead of conditionally omitting a cue.
        prior: Prior probability of the positive class.

    Returns:
        The fused probability and the breakdown that produced it.
    """
    breakdown = ScoreBreakdown()
    total = to_logit(prior)
    breakdown.add("prior", to_logit(prior), 1.0)

    for name, (probability, weight) in evidence.items():
        if weight <= 0:
            continue
        # Each cue contributes only its *departure* from the prior, so a
        # neutral cue is genuinely neutral no matter how many are present.
        delta = to_logit(probability) - to_logit(prior)
        contribution = weight * delta
        total += contribution
        breakdown.add(name, contribution, weight)

    breakdown.fused_logit = total
    return to_probability(total), breakdown


def confidence_from_logit(logit: float, scale: float = 2.0) -> float:
    """Map a fused logit to a 0-1 confidence (distance from indecision)."""
    return float(np.clip(abs(logit) / max(scale, EPS), 0.0, 1.0))


def agreement_confidence(breakdown: ScoreBreakdown) -> float:
    """How much the cues agreed, ignoring the prior.

    Returns 1.0 when every contributing cue points the same way and 0.0 when
    they cancel out exactly. Reported alongside the posterior so that a
    confident-looking probability built on contradictory cues is visible.
    """
    contributions = [v for k, v in breakdown.cues.items() if k != "prior"]
    if not contributions:
        return 0.0
    magnitude = sum(abs(c) for c in contributions)
    if magnitude < EPS:
        return 0.0
    return float(abs(sum(contributions)) / magnitude)


class LinearProbe:
    """A logistic-regression head over VLM embeddings.

    This is the cheap-training escape hatch. Fitting it needs only a few dozen
    labelled crops and finishes in under a second on a CPU, which means a user
    can adapt the sex classifier to their own cultivar and lighting without ever
    touching a GPU - while the zero-shot cues keep working if they never do.
    """

    def __init__(self, weights: np.ndarray, bias: float, classes: list[str]):
        self.weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        self.bias = float(bias)
        self.classes = list(classes)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Positive-class probability for each row of ``features``."""
        features = np.atleast_2d(np.asarray(features, dtype=np.float32))
        logits = features @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-logits))

    def save(self, path: str) -> None:
        np.savez(path, weights=self.weights, bias=self.bias, classes=np.array(self.classes))

    @classmethod
    def load(cls, path: str) -> "LinearProbe":
        data = np.load(path, allow_pickle=False)
        return cls(data["weights"], float(data["bias"]), [str(c) for c in data["classes"]])

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, classes: list[str],
            epochs: int = 400, learning_rate: float = 0.5,
            l2: float = 1e-3) -> "LinearProbe":
        """Fit by full-batch gradient descent.

        Deliberately dependency-free: adding scikit-learn to the rover image for
        one logistic regression is not a trade worth making.
        """
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32).reshape(-1)
        n, dim = features.shape
        weights = np.zeros(dim, dtype=np.float32)
        bias = 0.0
        for _ in range(epochs):
            logits = features @ weights + bias
            predictions = 1.0 / (1.0 + np.exp(-logits))
            error = predictions - labels
            grad_w = features.T @ error / n + l2 * weights
            grad_b = float(error.mean())
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b
        return cls(weights, bias, classes)
