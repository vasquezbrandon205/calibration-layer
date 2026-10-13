"""calibration-layer: a measured confidence layer for model answers.

Ask the model the same question several ways. If its answers agree, trust it.
If they don't, don't. This module turns that intuition into a calibrated
probability, fitted on sealed first-person data (n=29) and verified
out-of-sample (n=6).

Fitted model (2026-10-13):
    P(correct) = sigmoid(8.5 * consistency + 5.0 * (mean_confidence / 100) - 8.5)

Consistency dominates: on the fit data, correct answers averaged 0.97
consistency vs 0.42 for wrong answers. Verbalized confidence alone is weaker
(consistent with Müller et al. 2026).

Known limitation (v0.1): correlated systematic errors defeat consistency —
if every sample makes the SAME mistake, agreement means nothing. Vary the
method across samples (see calibrate() docs).
"""

import math
from collections import Counter

_W_CONSISTENCY = 8.5
_W_CONFIDENCE = 5.0
_BIAS = -8.5


def _sigmoid(z):
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def p_correct(consistency, mean_confidence):
    """Calibrated probability the modal answer is correct.

    Args:
        consistency: fraction of samples agreeing with the modal answer (0..1).
        mean_confidence: mean verbalized confidence across samples (0..100).
    """
    if not 0.0 <= consistency <= 1.0:
        raise ValueError("consistency must be in [0, 1]")
    z = _W_CONSISTENCY * consistency + _W_CONFIDENCE * (mean_confidence / 100.0) + _BIAS
    return _sigmoid(z)


def calibrate(samples):
    """Run the calibration protocol on N samples.

    Args:
        samples: list of (answer, confidence) tuples, one per independent
            attempt. Attempts MUST use different methods/derivations where
            possible — correlated methods hide systematic errors.

    Returns:
        (modal_answer, p): the most common answer and its calibrated
        probability of being correct.
    """
    if len(samples) < 3:
        raise ValueError("need at least 3 samples for a meaningful consistency signal")
    answers = [a for a, _ in samples]
    confs = [c for _, c in samples]
    modal, topn = Counter(answers).most_common(1)[0]
    consistency = topn / len(samples)
    mean_conf = sum(confs) / len(confs)
    return modal, p_correct(consistency, mean_conf)


def should_trust(samples, threshold=0.5):
    """Convenience: (modal_answer, p, trust) with an explicit decision."""
    modal, p = calibrate(samples)
    return modal, p, p >= threshold
