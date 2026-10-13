# calibration-layer

**A measured confidence layer for AI answers — built on sealed first-person data, not vibes.**

AI assistants can't reliably tell you when they're wrong. Verbalized confidence ("I'm 90% sure") is systematically biased. This project ships a calibrated alternative: sample the question several ways, measure how much the answers *agree*, and map that to a real probability — fitted on sealed measurements, verified out-of-sample.

P(correct) = sigmoid(8.5 · consistency + 5.0 · (confidence/100) − 8.5)

## The core finding

**Consistency dominates.** Across 29 sealed observations, correct answers averaged **0.97** answer-agreement vs **0.42** for wrong ones — a cliff, not a slope. Asking the model five different ways and counting agreement beats asking it how sure it is. This independently confirms Müller et al. (2026) with first-person sealed data.

## Quickstart

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

See examples/quickstart.py for a fuller walkthrough.

## How it was built (sealed-truth methodology)

1. **Measured the baseline** — 90 blind samples across 15 questions. Found the data had no errors in it, so nothing could be learned.
2. **Hunted the error region** — sealed 15-question battery of increasing difficulty, answers locked before sampling. Mapped the frontier: mastery → error region → infeasibility → honest guessing.
3. **Fit the function** — logistic regression on (consistency, confidence) → correctness, n=29.
4. **Verified out-of-sample** — 6 fresh sealed questions, predictions locked before unsealing: 4/6 directional hits.

Full mission report: docs/REPORT.md.

## Honest limitations (v0.1)

- **Correlated errors defeat consistency.** If every attempt makes the same mistake, agreement means nothing. Mitigation: vary the method across samples; v0.2 will add a method-diversity term.
- **Small n.** Fit on 29 questions, verified on 6.
- **Validated domain:** computational questions + general knowledge.

## What's here

- calibration/layer.py — the layer: p_correct, calibrate, should_trust
- data/fit_data.json — 29 calibration fit points
- data/verification_data.json — 6 out-of-sample verification points + predictions
- docs/REPORT.md — full mission report with methodology
- examples/quickstart.py — worked example

## License

Apache 2.0 — use it, build on it, commercialize it. See LICENSE.
