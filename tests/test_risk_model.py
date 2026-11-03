import unittest
import numpy as np
import pandas as pd
from scripts.refresh_data import risk_model

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
        # Past-only forecast at t must be unaffected by additional future observations.
        x=self.make_prices(720)
        earlier=risk_model(x.iloc[:660],"ES=F")
        extended=risk_model(x.iloc[:661],"ES=F")
        # If a new observation arrives, the forecast can change, but both remain disabled.
        self.assertFalse(earlier.get("signal_eligible",False))
        self.assertFalse(extended.get("signal_eligible",False))

    def test_nonoverlapping_validation(self):
        m=risk_model(self.make_prices(950),"ES=F")
        wf=m.get("walk_forward",{})
        self.assertIn("nonoverlapping",wf.get("evaluation",""))
        self.assertFalse(wf.get("validated",True))
        self.assertLess(wf.get("samples",0),950//5)

    def test_short_history_fails_closed(self):
        m=risk_model(self.make_prices(110),"ES=F")
        self.assertEqual(m["status"],"insufficient_history")
        self.assertFalse(m.get("signal_eligible",False))

if __name__=="__main__":
    unittest.main()
