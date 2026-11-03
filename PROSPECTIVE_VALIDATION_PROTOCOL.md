# MTM prospective validation protocol

Frozen research-model identifier: `mtm-calibrated-v1-2026-11-03`.
Prospective start: **after 2026-11-03**. The initial ledger is intentionally empty: historical records must not be relabeled as genuinely prospective.

The GitHub Actions refresh computes daily public ES/NQ prices and probabilities, then appends a forecast no more than once per five observed sessions for each market. Each record includes market, forecast date, reference close, downside threshold, calibrated probability, baseline probability and model identifier. A record is scored only after five subsequent daily market observations. Existing forecasts are not recomputed on later runs; outcome and Brier scores are filled when matured.

This is an **operational prospective log**, not a guarantee of immutable tamper-proof storage. The repository is editable, and independent verification of the Yahoo Finance continuous-futures roll methodology remains outstanding. Do not modify model hyperparameters during prospective collection; any change requires a new model version and separate ledger.

Validation success requires at least 100 matured, nonoverlapping prospective forecasts per market, a predeclared positive improvement against the same-date historical base-rate benchmark with confidence interval excluding zero, acceptable reliability calibration and verified data provenance. Trading costs/slippage and independent roll-adjusted futures data must be assessed before live trading. **Signals remain disabled**. Credit Pulse is out of scope.
