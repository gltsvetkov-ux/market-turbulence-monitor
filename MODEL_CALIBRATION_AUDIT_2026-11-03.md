# MTM probability calibration audit — 2026-11-03

## Procedure
Five years of Yahoo ES=F and NQ=F daily close history (1,258 observations per market). Existing raw forecasts use expanding past-only five-day event labels and a 5-day outcome embargo. Calibration uses **only previously matured, nonoverlapping out-of-sample forecasts**. It starts after 40 such observations, estimates mean realized-minus-predicted probability, shrinks the correction by n/(n+40), clips the correction to +/-15 percentage points and final probability to [0.001,0.999]. Parameters were chosen before reviewing the following comparison; the results are diagnostic, not independent hyperparameter validation.

## Matched-period backtest
| Metric | ES | NQ |
|---|---:|---:|
| Calibration evaluation anchors | 166 | 168 |
| Raw Brier | 0.1527445 | 0.1410149 |
| Calibrated Brier | 0.1517962 | 0.1408214 |
| Past-only base-rate Brier | 0.1511723 | 0.1416089 |
| Raw forecast minus observed event frequency | +4.64 pp | +4.55 pp |
| Calibrated forecast minus observed frequency | +0.41 pp | -1.93 pp |
| Calibrated improvement over baseline, paired 3000-bootstrap 95% CI | [-0.00845,+0.00691] | [-0.00901,+0.00960] |

## Interpretation
Calibration improves the raw Brier score and reduces aggregate probability bias in both markets. ES still trails the rolling historical baseline on matched observations. NQ's apparent improvement over the baseline is statistically inconclusive; both confidence intervals include zero. The bootstrap treats nonoverlapping evaluation windows as independently resampled, although market regimes can remain dependent. No transaction-cost, external roll-adjusted futures, or genuinely independent holdout validation is available.

## Deployment and safety
Source and deterministic tests updated in GitHub. The scheduled GitHub Actions refresh is responsible for writing a fresh calibrated data.json snapshot; until it does, the public dashboard may still display the preceding snapshot. Signals remain disabled regardless of in-sample or walk-forward score. Credit Pulse untouched.
