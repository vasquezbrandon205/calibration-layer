"""calibration-layer v0.3: tripwire verification + adaptive-K.

v0.2's remaining gap (pinpointed): every signal still came from the model's
own outputs. The field's answer (cascades, Oct 2026): route to verifiers that
are NOT language models — components that fail for different reasons.

v0.3 adds a verifier plugin registry. The first plugin is exact
recomputation for modular arithmetic: the model reasons fallibly, the
verifier computes infallibly — implementation diversity, the strongest
decorrelation. Retrospective proof: the V6 killing-case wrong answer
(1147912 vs true 146440) fails exact recomputation deterministically.

Bayesian update on pass (probabilistic verifiers):
    P_post = P_prior / (P_prior + (1 - P_prior) * fpr)
Exact verifier (fpr=0): pass -> P=1.0, fail -> P=0.0. Decisive.

Also adds Adaptive-K (after Yu et al., arXiv:2609.38829, Oct 2026): a 4-path
pilot decides whether more samples are worth the cost. Unanimous pilot ->
stop; split pilot -> extend.
"""

import math
from collections import Counter
from math import prod

try:
    from .layer_v2 import p_correct, calibrate_v2, ASSIGNABLE_METHODS
except ImportError:  # standalone script use
    from layer_v2 import p_correct, calibrate_v2, ASSIGNABLE_METHODS


# ---------------------------------------------------------------------------
# Verifier plugin registry
# ---------------------------------------------------------------------------

VERIFIERS = {}


def verifier(domain):
    def deco(fn):
        VERIFIERS[domain] = fn
        return fn
    return deco


@verifier("modular_arithmetic")
def mod_tripwire(answer, params):
    """Exact-recomputation tripwire for base^exp mod m claims.

    The model derived the answer by fallible reasoning; the verifier
    recomputes with exact machine arithmetic — O(log exp), infallible.
    This is implementation diversity: reasoner vs calculator fail
    differently, which is the strongest form of decorrelation.

    NOTE (integrity): an earlier version checked answer mod p against
    pow(base, exp, p) for small primes p not dividing m — mathematically
    INVALID, caught by self-test (it rejected the true answer). The only
    valid shortcut is exact recomputation. Probabilistic tripwires
    (Freivalds-style) belong in future plugins where exact recompute is
    genuinely expensive (e.g. matrix products).

    Args:
        answer: the model's claimed answer (int)
        params: (base, exp, mod) tuple

    Returns:
        dict(passed=bool, false_pass_rate=0.0, detail=str)
    """
    base, exp, mod = params
    if not isinstance(answer, int) or not (0 <= answer < mod):
        return {"passed": False, "false_pass_rate": 0.0,
                "detail": "answer out of range [0, mod)"}
    true_val = pow(base, exp, mod)
    return {"passed": (answer == true_val), "false_pass_rate": 0.0,
            "detail": "exact recomputation"}


def apply_tripwire(p_prior, trip):
    """Fold a tripwire result into P(correct).

    Exact verifier (fpr=0): decisive — pass -> 1.0, fail -> 0.0.
    Probabilistic verifier (fpr>0): Bayesian update
        P_post = P_prior / (P_prior + (1 - P_prior) * fpr).
    """
    if trip["passed"]:
        fpr = trip["false_pass_rate"]
        if fpr == 0.0:
            return 1.0
        return p_prior / (p_prior + (1.0 - p_prior) * fpr)
    return 0.0


# ---------------------------------------------------------------------------
# v0.3 calibration: v0.2 + optional tripwire
# ---------------------------------------------------------------------------

def calibrate_v3(samples, domain=None, domain_params=None):
    """Full v0.3 pipeline.

    Args:
        samples: list of (answer, confidence, method) — see layer_v2
        domain: verifier domain key (e.g. "modular_arithmetic"), or None
        domain_params: params for the verifier (e.g. (base, exp, mod))

    Returns:
        (modal_answer, p, report)
    """
    modal, p, report = calibrate_v2(samples)
    report = dict(report)
    report["version"] = "0.3"
    report["tripwire"] = None

    if p is None:
        return modal, None, report  # diversity abstention stands

    if domain and domain in VERIFIERS:
        trip = VERIFIERS[domain](modal, domain_params)
        p = apply_tripwire(p, trip)
        report["tripwire"] = {
            "domain": domain,
            "passed": trip["passed"],
            "detail": trip.get("detail", ""),
            "false_pass_rate": trip["false_pass_rate"],
        }
        report["trust"] = p >= 0.5
    return modal, p, report


# ---------------------------------------------------------------------------
# Adaptive-K: pilot of 4, extend only on disagreement
# ---------------------------------------------------------------------------

def adaptive_k_decision(pilot_answers):
    """After a 4-sample pilot (one per method family): unanimous -> stop,
    any split -> extend sampling. Returns 'stop' or 'extend'."""
    if len(set(pilot_answers)) == 1:
        return "stop"
    return "extend"
