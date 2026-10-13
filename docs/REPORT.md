# Calibration Layer — Mission Report (2026-10-13)

## Mission
Apply the pinpoint tool to an unsolved industry problem: LLM calibration.
Build the layer nobody has — a measured mapping from observable signals to P(correct).

## Phase 1 — TEST
Mined 5 N-sampling pilots (90 samples, 15 questions). Finding: 15/15 modal-correct, consistency at 1.00 on 14/15. The data had no errors — nothing to learn calibration from.

## Phase 2 — IDENTIFY
Pinpointed gap: zero error-region data. Fix: extend the Ignorance Map past L3 until errors occur.

## Phase 3 — FIX
- Sealed 15-question battery (L4–L8), answers locked before sampling.
- Blind N-sampling, strict mental-math discipline. Two contaminations caught and burned: a sampler cross-checking with Python pow(), and a sampler glimpsing prior answers (anchoring risk).
- Boundary discovery: L7 = infeasibility frontier (one genuine sample took ~75 min of hand arithmetic; sampler refused to fabricate the rest). L8 = pure guessing (confidence 2.0).
- Unsealed 14 questions. Fit (n=29 with pilots): P(correct) = sigmoid(8.5*consistency + 5.0*(conf/100) − 8.5). Consistency dominates (correct mean 0.97 vs wrong mean 0.42).

## Phase 4 — VERIFY (out-of-sample, n=6, predictions locked pre-unseal)
4/6 directional hits. The two misses are the most valuable data:
- V1 (pred 0.083, actually correct): low-consistency modal (2/5) was right — under-trusted a shaky plurality.
- V6 (pred 0.979, actually wrong): all samples agreed and all were wrong — correlated systematic error defeats consistency. This is the layer's boundary condition and the v0.2 roadmap (method-diversity term).

## Honest assessment
The layer works directionally; its core claim (consistency dominates, confirming Müller 2026 with first-person sealed data) survived fresh data. v0.1 ships with the correlated-error limitation documented, not hidden.
