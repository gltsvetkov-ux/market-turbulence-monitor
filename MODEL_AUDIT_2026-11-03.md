# MTM five-year independent model audit — 2026-11-03

## Scope
ES=F and NQ=F, 1,258 Yahoo Finance daily closing prices each, 2021-11-03 through 2026-11-03. Five-session downside events: ES -2%, NQ -3%, measured from a signal close to the minimum of the following five closes. Expanding past-only training with completed-label embargo. One nonoverlapping evaluation anchor per five sessions.

## Reproduced scores (full precision)
| Metric | ES | NQ |
|---|---:|---:|
| Test anchors | 207 | 209 |
| Brier model | 0.1597103 | 0.1474969 |
| Brier rolling base rate | 0.1601448 | 0.1489905 |
| Improvement (base minus model) | 0.0004345 | 0.0014936 |
| Paired bootstrap 95% CI, 3000 draws, reproducible LCG | [-0.00423, +0.00493] | [-0.00251, +0.00554] |
| Observed event frequency | 19.81% | 17.22% |
| Mean predicted event frequency | 25.25% | 24.31% |
| Recent 30% Brier (model / base) | 0.12113 / 0.11889 | 0.10450 / 0.10615 |

## Findings
- Both apparent improvements are statistically inconclusive (CI includes zero).
- Mean predicted probabilities exceed realized event rates by ~5.44 percentage points ES and ~7.09 points NQ in the evaluated anchors: evidence of aggregate overforecasting; calibration needs attention.
- In the last 30% of anchors, ES underperforms the baseline.
- No proof of out-of-sample trading profitability; costs and slippage have not been tested.
- Yahoo continuous futures contract-roll adjustment is not independently verified.
- Bootstrap resamples nonoverlapping windows independently; market-regime dependence may still understate uncertainty. This is a diagnostic, not a formal performance guarantee.

## Operational decision
Keep `signal_eligible=false` and `signal=disabled_unvalidated`. Model may display research probabilities with explicit limitations. Do not connect live trading or Credit Pulse.
