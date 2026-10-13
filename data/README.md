# Data dictionary

## fit_data.json
29 calibration fit points. Each point:
- id: question identifier
- source: "n-sampling pilot" (general knowledge) or "sealed battery L4-L8" (modular arithmetic)
- consistency: fraction of blind samples agreeing with the modal answer (0..1)
- mean_confidence: mean verbalized confidence across samples (0..100)
- modal_correct: whether the modal answer matched sealed ground truth

## verification_data.json
6 out-of-sample verification questions. Predictions were locked BEFORE unsealing.
- predicted_p_correct: model output before seeing ground truth
- modal_correct: ground-truth outcome
- Result: 4/6 directional hits.
