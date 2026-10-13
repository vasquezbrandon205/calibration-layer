"""calibration-layer v0.2: family-level consistency.

v0.1's failure (V6): all samples agreed and all were wrong — one method,
one shared blind spot, counted as n independent votes. v0.2 fixes the unit
of independence: the method family, not the sample.

Rule: the layer counts one vote per method family. Five samples from one
method = one vote. Agreement across genuinely different methods is the
signal; agreement within a method is slip-checking only.

Design note (honest): the fitted coefficients come from individual-level
data (n=29). Applying them at family level is an extrapolation, flagged
below and validated against the V6 killing case. Method labels are
caller-provided; the mechanism fails safe (fake diversity collapses to
one family -> abstain), but for adversarial settings the layer should
ASSIGN methods rather than trust labels (see assign_methods).
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
    if not 0.0 <= consistency <= 1.0:
        raise ValueError("consistency must be in [0, 1]")
    z = _W_CONSISTENCY * consistency + _W_CONFIDENCE * (mean_confidence / 100.0) + _BIAS
    return _sigmoid(z)


def calibrate_v2(samples):
    """Calibrate with method-family independence.

    Args:
        samples: list of (answer, confidence, method) tuples. `method` is a
            family label like "raw-lsb", "crt", "fermat", "msb", "split-exp".
            Use genuinely different derivations per family.

    Returns:
        (modal_answer, p, report): p is None when the layer abstains
        (fewer than 2 method families — insufficient independent signal).
    """
    if not samples:
        raise ValueError("no samples")
    families = {}
    for ans, conf, method in samples:
        families.setdefault(method, []).append((ans, conf))

    n_fam = len(families)
    all_answers = [a for a, _, _ in samples]
    modal_all = Counter(all_answers).most_common(1)[0][0]

    if n_fam < 2:
        return modal_all, None, {
            "trust": False,
            "reason": "insufficient method diversity: "
                      f"{n_fam} family(ies), need >= 2 for independent signal",
            "families": n_fam,
            "extrapolation": False,
        }

    fam_votes, fam_confs = [], []
    for method, ss in families.items():
        fa = Counter(a for a, _ in ss).most_common(1)[0][0]
        fam_votes.append(fa)
        fam_confs.append(sum(c for _, c in ss) / len(ss))

    modal_fam, topn = Counter(fam_votes).most_common(1)[0]
    cons_fam = topn / n_fam
    mconf_fam = sum(fam_confs) / len(fam_confs)
    p = p_correct(cons_fam, mconf_fam)
    return modal_fam, p, {
        "trust": p >= 0.5,
        "families": n_fam,
        "family_consistency": round(cons_fam, 3),
        "family_mean_conf": round(mconf_fam, 1),
        # Honest flag: coefficients fit at individual level, applied at family level
        "extrapolation": True,
    }


# Method families the layer can ASSIGN (v0.2, computational domain).
# For adversarial settings, assign these instead of trusting caller labels.
ASSIGNABLE_METHODS = [
    "raw-lsb",      # binary square-and-multiply, LSB first, full modulus
    "raw-msb",      # MSB-first bit scan, different multiplication ordering
    "crt",          # factor modulus, solve per factor, combine
    "fermat",       # exponent reduction via totient/Carmichael
    "split-exp",    # exponent decomposition a*b+c, independent regrouping
]


def assign_methods(n):
    """Assign n distinct method families round-robin. Fails safe: if n >
    available families, families repeat and the diversity count reflects it."""
    return [ASSIGNABLE_METHODS[i % len(ASSIGNABLE_METHODS)] for i in range(n)]
