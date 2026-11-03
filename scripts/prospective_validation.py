"""Append-only prospective MTM forecast ledger. No Credit Pulse or brokerage access."""
import json
from pathlib import Path
from datetime import date
import math

FREEZE_DATE = "2026-11-03"
MODEL_VERSION = "mtm-calibrated-v1-2026-11-03"
LEDGER = Path("prospective_forecasts.json")

def update(snapshot, ledger):
    if not isinstance(ledger, dict) or ledger.get("model_version") != MODEL_VERSION:
        raise ValueError("Ledger version mismatch: refusing to rewrite historical forecasts")
    records = ledger["forecasts"]
    if not isinstance(records,list): raise ValueError("Invalid ledger")
    ids = [(r["market"],r["forecast_date"]) for r in records]
    if len(ids)!=len(set(ids)): raise ValueError("Duplicate forecast keys")
    # Grade only forecasts whose full five *subsequent observed sessions* exist.
    for r in records:
        if r.get("outcome") is not None: continue
        history = snapshot["markets"][r["market"]]["history"]
        dates = [h["date"] for h in history]
        if r["forecast_date"] not in dates: continue
        i=dates.index(r["forecast_date"])
        if len(history) <= i+5: continue
        future=[float(x["close"]) for x in history[i+1:i+6]]
        outcome=int(min(future)/float(r["reference_close"])-1 <= -float(r["threshold_pct"])/100)
        r["outcome"]=outcome
        r["outcome_matured_on"]=history[i+5]["date"]
        r["brier"] = round((float(r["calibrated_probability"])-outcome)**2,8)
        r["baseline_brier"] = round((float(r["baseline_probability"])-outcome)**2,8)
    for market in ("ES","NQ"):
        m=snapshot["markets"][market]
        day=m["asof"]
        if day<=FREEZE_DATE or (market,day) in ids: continue
        # Nonoverlapping forecasts: >= 5 subsequent trading sessions since prior anchor.
        previous=[r for r in records if r["market"]==market]
        history=m["history"]
        dates=[h["date"] for h in history]
        if previous:
            last=max(r["forecast_date"] for r in previous)
            if last not in dates or len(dates)-1-dates.index(last)<5: continue
        risk=m["risk_model"]
        if risk.get("status")!="experimental" or not risk.get("calibration"): continue
        p=float(risk["calibrated_probability_pct"])/100
        b=float(risk["base_rate_pct"])/100
        if not (math.isfinite(p) and math.isfinite(b) and 0<=p<=1 and 0<=b<=1):
            raise ValueError("Invalid forecast probability")
        records.append({"market":market,"forecast_date":day,"model_version":MODEL_VERSION,
            "reference_close":float(m["close"]),"threshold_pct":float(risk["threshold_pct"]),
            "calibrated_probability":p,"baseline_probability":b,
            "outcome":None,"outcome_matured_on":None})
    return ledger

def main():
    snapshot=json.loads(Path("data.json").read_text())
    ledger=json.loads(LEDGER.read_text())
    updated=update(snapshot,ledger)
    LEDGER.write_text(json.dumps(updated,indent=2,ensure_ascii=False)+"\n")
    print("Prospective records:",len(updated["forecasts"]),"graded:",sum(x["outcome"] is not None for x in updated["forecasts"]))

if __name__=="__main__":main()
