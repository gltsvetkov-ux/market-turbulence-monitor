import unittest
from copy import deepcopy
from scripts.prospective_validation import update, MODEL_VERSION

class ProspectiveValidationTests(unittest.TestCase):
    def snapshot(self, days):
        markets={}
        for market,threshold in (("ES",2),("NQ",3)):
            history=[{"date":day,"close":100.0} for day in days]
            markets[market]={"history":history,"asof":days[-1],"close":100.0,
                "risk_model":{"status":"experimental","calibration":{"validated":False},
                  "calibrated_probability_pct":15.0,"base_rate_pct":20.0,"threshold_pct":threshold}}
        return {"markets":markets}

    def ledger(self):
        return {"model_version":MODEL_VERSION,"forecasts":[]}

    def test_first_forecast_is_frozen_and_no_duplicate(self):
        days=["2026-11-04","2026-11-05"]
        ledger=update(self.snapshot(days),self.ledger())
        self.assertEqual(len(ledger["forecasts"]),2)
        original=deepcopy(ledger)
        update(self.snapshot(days),ledger)
        self.assertEqual(ledger,original)
        self.assertTrue(all(r["outcome"] is None for r in ledger["forecasts"]))

    def test_grading_waits_for_five_subsequent_sessions(self):
        days=["2026-11-04","2026-11-05","2026-11-06","2026-11-09","2026-11-10"]
        ledger=update(self.snapshot(days[:1]),self.ledger())
        update(self.snapshot(days),ledger)
        self.assertTrue(all(r["outcome"] is None for r in ledger["forecasts"]))

    def test_matured_forecast_grades_once(self):
        days=["2026-11-04","2026-11-05","2026-11-06","2026-11-09","2026-11-10","2026-11-11"]
        ledger=update(self.snapshot(days[:1]),self.ledger())
        snapshot=self.snapshot(days)
        snapshot["markets"]["ES"]["history"][3]["close"]=97.0
        update(snapshot,ledger)
        es=next(r for r in ledger["forecasts"] if r["market"]=="ES")
        nq=next(r for r in ledger["forecasts"] if r["market"]=="NQ")
        self.assertEqual(es["outcome"],1)
        self.assertEqual(nq["outcome"],0)
        self.assertEqual(es["outcome_matured_on"],"2026-11-11")
        saved=deepcopy(es)
        snapshot["markets"]["ES"]["history"][3]["close"]=100.0
        update(snapshot,ledger)
        self.assertEqual(es,saved)

    def test_wrong_version_rejected(self):
        with self.assertRaises(ValueError):
            update(self.snapshot(["2026-11-04"]),{"model_version":"other","forecasts":[]})

if __name__=="__main__":
    unittest.main()
