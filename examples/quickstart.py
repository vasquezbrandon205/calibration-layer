"""Worked example: calibrate an answer from 5 independent attempts."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration.layer import calibrate, should_trust

samples = [
    (421, 80),
    (421, 75),
    (421, 85),
    (389, 60),
    (421, 70),
]

answer, p = calibrate(samples)
print(f"answer={answer}  P(correct)={p:.3f}")

answer, p, trust = should_trust(samples, threshold=0.5)
print(f"trust this answer: {trust}")

bad = [(1, 10), (2, 15), (3, 12)]
print(should_trust(bad))
