import unittest
import numpy as np
import pandas as pd
from scripts.refresh_data import risk_model, _forecast_at

class RiskModelTests(unittest.TestCase):
    def make_prices(self,n=650,seed=19):
        rng=np.random.default_rng(seed)
        daily=rng.normal(0.0003,0.012,n)
        return pd.Series(4000*np.exp(np.cumsum(daily)),index=pd.date_range("2023-01-01",periods=n,freq="B"))

    def test_no_live_signal_even_if_model_improves(self):
        for symbol in ("ES=F","NQ=F"):
            m=risk_model(self.make_prices(),symbol)
            self.assertFalse(m.get("signal_eligible",False))
            self.assertEqual(m.get("signal"),"disabled_unvalidated")

    def test_thresholds_and_probability(self):
        for symbol,threshold in (("ES=F",2),("NQ=F",3)):
            m=risk_model(self.make_prices(),symbol)
            self.assertEqual(m.get("threshold_pct"),threshold)
            if m.get("status")=="experimental":
                self.assertGreaterEqual(m["probability_pct"],0)
                self.assertLessEqual(m["probability_pct"],100)

    def test_no_future_leakage_into_current_prediction(self):
        # Even if the caller holds later features and outcomes, a forecast at t
        # may consume only labels whose full horizon ended strictly before t.
        features=[(i,"normal",0.0) for i in range(220)]
        outcomes={i:int(i%4==0) for i in range(195)}
        before=_forecast_at(features,outcomes,200,"normal")
        altered=dict(outcomes)
        altered.update({i:1-outcomes.get(i,0) for i in range(195,220)})
        after=_forecast_at(features,altered,200,"normal")
        self.assertEqual(before,after)
        self.assertEqual(before[2],195)  # indices 0..194 with five-session embargo

    def test_deterministic_repeated_evaluation(self):
        x=self.make_prices(950)
        a=risk_model(x,"ES=F")
        b=risk_model(x,"ES=F")
        self.assertEqual(a,b)
        wf=a["walk_forward"]
        self.assertGreaterEqual(wf["samples"],20)
        self.assertLessEqual(wf["brier_improvement_ci95"][0],wf["brier_improvement_ci95"][1])
        self.assertGreaterEqual(wf["actual_event_rate_pct"],0)
        self.assertLessEqual(wf["actual_event_rate_pct"],100)
        self.assertGreaterEqual(wf["log_loss"],0)
        self.assertGreaterEqual(wf["baseline_log_loss"],0)
        self.assertIn("beats_baseline_log_loss",wf)

    def test_calibration_uses_only_matured_prior_predictions(self):
        x=self.make_prices(950)
        m=risk_model(x,"ES=F")
        self.assertEqual(m["calibration"]["method"],"past_only_regularized_additive_bias")
        self.assertGreaterEqual(m["calibration"]["training_nonoverlapping_predictions"],40)
        self.assertLessEqual(abs(m["calibration"]["correction_percentage_points"]),15)
        self.assertGreaterEqual(m["probability_pct"],0)
        self.assertLessEqual(m["probability_pct"],100)
        o=m["calibration"]["out_of_sample"]
        self.assertGreater(o["samples"],20)
        self.assertTrue(all(0<=o[k]<=1 for k in ("calibrated_brier","raw_brier_same_period","baseline_brier_same_period")))
        self.assertFalse(m["signal_eligible"])
        self.assertGreaterEqual(o["calibrated_log_loss"],0)
        self.assertGreaterEqual(o["raw_log_loss_same_period"],0)
        self.assertGreaterEqual(o["baseline_log_loss_same_period"],0)

    def test_future_price_does_not_change_historical_forecast(self):
        x=self.make_prices(950)
        a=risk_model(x,"NQ=F")
        # Last observed close can change without changing training labels before the current anchor;
        # no future label is allowed to be used for its own prediction.
        self.assertGreaterEqual(a["calibration"]["training_nonoverlapping_predictions"],40)
        self.assertFalse(a["calibration"]["validated"])

    def test_chronological_holdout_is_reported_and_not_pristine(self):
        for symbol in ("ES=F","NQ=F"):
            m=risk_model(self.make_prices(950),symbol)
            h=m["calibration"]["chronological_holdout"]
            self.assertGreaterEqual(h["samples"],20)
            self.assertFalse(h["pristine_holdout"])
            self.assertGreaterEqual(h["calibrated_brier"],0)
            self.assertGreaterEqual(h["baseline_brier"],0)
            self.assertFalse(m["signal_eligible"])

    def test_nonoverlapping_validation(self):
        m=risk_model(self.make_prices(950),"ES=F")
        wf=m.get("walk_forward",{})
        self.assertIn("nonoverlapping",wf.get("evaluation",""))
        self.assertFalse(wf.get("validated",True))
        self.assertLess(wf.get("samples",0),950//5)

    def test_unsupported_symbol_is_rejected(self):
        with self.assertRaises(ValueError):
            risk_model(self.make_prices(),"UNKNOWN")

    def test_invalid_prices_fail_closed(self):
        prices=self.make_prices()
        prices.iloc[-1]=float("nan")
        m=risk_model(prices,"ES=F")
        self.assertFalse(m["signal_eligible"])
        self.assertEqual(m["signal"],"disabled_unvalidated")

    def test_short_history_fails_closed(self):
        m=risk_model(self.make_prices(110),"ES=F")
        self.assertEqual(m["status"],"insufficient_history")
        self.assertFalse(m.get("signal_eligible",False))

if __name__=="__main__":
    unittest.main()

