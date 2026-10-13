# calibration-layer

**A measured confidence layer for AI answers — built on sealed first-person data, not vibes.**

AI assistants can't reliably tell you when they're wrong. Verbalized confidence ("I'm 90% sure") is systematically biased. This project ships a calibrated alternative: sample the question several ways, measure how much the answers *agree*, and map that to a real probability — fitted on sealed measurements, verified out-of-sample.

```
P(correct) = sigmoid(8.5 · consistency + 5.0 · (confidence/100) − 8.5)
```

## The core finding

**Consistency dominates.** Across 29 sealed observations, correct answers averaged **0.97** answer-agreement vs **0.42** for wrong ones — a cliff, not a slope. Asking the model five different ways and counting agreement beats asking it how sure it is. This independently confirms Müller et al. (2026) with first-person sealed data.

## Quickstart

```python
from calibration.layer import calibrate

samples = [
    (421, 80),  # (answer, confidence) from attempt 1
    (421, 75),  # attempt 2 — use a DIFFERENT method each time
    (421, 85),  # attempt 3
    (389, 60),  # attempt 4 — one disagrees
    (421, 70),  # attempt 5
]
answer, p = calibrate(samples)
print(answer, round(p, 3))  # 421 0.881
```

See `examples/quickstart.py` for a fuller walkthrough.

## How it was built (sealed-truth methodology)

1. **Measured the baseline** — 90 blind samples across 15 questions. Found the data had no errors in it, so nothing could be learned. (You can't learn what "wrong" looks like without wrong.)
2. **Hunted the error region** — built a sealed 15-question battery of increasing difficulty (hand-computed modular arithmetic, L4–L8), answers locked before sampling. Mapped the frontier: mastery → error region → infeasibility → honest guessing.
3. **Fit the function** — logistic regression on (consistency, confidence) → correctness, n=29.
4. **Verified out-of-sample** — 6 fresh sealed questions, predictions locked *before* unsealing: **4/6 directional hits**, including correctly flagging answers the model got wrong.

Full mission report: [`docs/REPORT.md`](docs/REPORT.md).

## Honest limitations (v0.1)

- **Correlated errors defeat consistency.** If every attempt makes the *same* mistake, agreement means nothing. Mitigation: vary the method across samples (required by the protocol); v0.2 will add a method-diversity term.
- **Small n.** Fit on 29 questions, verified on 6. Uncertainty is bimodal (confident or collapsed); middle probabilities are thin.
- **Validated domain:** computational questions + general knowledge. Broader domains need their own fits.

## What's here

| Path | Contents |
|---|---|
| `calibration/layer.py` | The layer: `p_correct`, `calibrate`, `should_trust` |
| `data/fit_data.json` | 29 calibration fit points |
| `data/verification_data.json` | 6 out-of-sample verification points + predictions |
| `docs/REPORT.md` | Full mission report with methodology |
| `examples/quickstart.py` | Worked example |

## License

Apache 2.0 — use it, build on it, commercialize it. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{vasquez2026calibration,
  author = {Vasquez, Brandon Manuel},
  title = {calibration-layer: a measured confidence layer for AI answers},
  year = {2026},
  url = {https://github.com/vasquezbrandon205/calibration-layer}
}
```

## v0.2 — family-level independence (2026-10-13)

v0.1 counted every sample as an independent vote. But five samples from one
method share one blind spot — correlated errors defeat consistency (the V6
case: 3/3 agreed, all wrong, P=0.979). v0.2 (`calibration/layer_v2.py`)
counts **one vote per method family** and **abstains** when fewer than two
families are present, instead of returning a confidently wrong number.

## v0.3 — tripwire verifiers (2026-10-13)

The strongest decorrelation is a verifier that isn't a model at all.
`calibration/layer_v3.py` adds a verifier plugin registry; the first plugin
does exact recomputation for modular arithmetic (decisive P=1.0 / P=0.0).
Validated: the V6 question re-sampled with 5 forced-diverse methods —
3 families (CRT, Fermat, split-exponent) independently converged on the true
answer 146440 while the two full-modulus attempts scattered with low
confidence. v0.2 scored P=0.443 (honest uncertainty); v0.3's tripwire
confirmed 146440 → P=1.0. Probe data: `data/v6_diverse_probe.json`.
