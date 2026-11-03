# Independent chronological validation audit — 2026-11-03

## Data and design
Independent reimplementation of the expanding past-only volatility-regime estimator and past-only regularized additive calibrator from 1,258 daily Yahoo Finance ES=F and NQ=F closes (2021-11-03 through 2026-11-03). Five-session downside event: ES -2%, NQ -3%; outcome known only after full 5-session horizon. One evaluation anchor per five sessions. Last 20% of available *calibrated* evaluation anchors used as a separate chronological test slice (34 per market). All training labels and calibrator residuals at a forecast time have fully matured before that time.

**Limitation:** This is an independent calculation and time-ordered slice, not a pristine never-seen holdout. Earlier model diagnostics and calibration decisions considered the full history. The 34 observations per market have limited statistical power. Futures roll-adjustment remains unverified.

## Results
| Metric | ES | NQ |
|---|---:|---:|
| Holdout anchors | 34 | 34 |
| First / last forecast | 2026-02-27 / 2026-10-23 | 2026-02-27 / 2026-10-23 |
| Raw Brier | 0.13432636 | 0.11587014 |
| Calibrated Brier | 0.13100538 | 0.11034942 |
| Historical past-only base-rate Brier | 0.12943339 | 0.10946521 |
| Observed event rate | 14.71% | 11.76% |
| Mean calibrated forecast | 16.93% | 13.94% |
| Paired IID bootstrap CI of baseline minus calibrated Brier (4000 reps) | [-0.01557,+0.01105] | [-0.01736,+0.01290] |

## Decision
Neither model beats the baseline on the time-ordered last-20% test. Confidence intervals include zero. Calibration improves raw model but not enough to outperform the historical base-rate benchmark. This does not establish a statistically significant trading edge or profit after transaction costs. Keep signal_eligible=false and signal=disabled_unvalidated. For genuinely independent validation, freeze model parameters and obtain prospective new observations or an independently sourced, verifiably rolled continuous futures dataset, then test before any re-tuning. Credit Pulse untouched.
