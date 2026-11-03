"""Public-data MTM snapshot. No brokerage credentials or Credit Pulse integration."""
import json, math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

def risk_model(close, symbol):
    """Expanding-window, strictly past-outcome conditional risk model.
    Event: within next five daily closes, loss of at least 2% ES / 3% NQ
    relative to signal-day close. No trades or direction forecast."""
    threshold = 0.02 if symbol == "ES=F" else 0.03
    horizon = 5
    values = close.to_numpy(dtype=float)
    returns = np.diff(np.log(values), prepend=np.nan)
    vol = pd.Series(returns).rolling(20).std().to_numpy() * math.sqrt(252)
    features = []
    for t in range(80, len(values)):
        if not np.isfinite(vol[t]): continue
        prior = vol[max(20,t-252):t]
        prior = prior[np.isfinite(prior)]
        if len(prior)<60: continue
        pct = float((prior <= vol[t]).mean())
        momentum = values[t]/values[t-20]-1
        bucket = ("high" if pct>=0.8 else "elevated" if pct>=0.6 else "normal")
        features.append((t,bucket,momentum))
    outcome = {}
    for t,bucket,momentum in features:
        if t+horizon < len(values):
            outcome[t] = int(np.min(values[t+1:t+horizon+1])/values[t]-1 <= -threshold)
    predictions=[]
    for t,bucket,momentum in features:
        if t not in outcome: continue
        eligible=[(i,b) for i,b,_ in features if i+horizon<t and i in outcome]
        if len(eligible)<120: continue
        matches=[outcome[i] for i,b in eligible if b==bucket]
        if len(matches)<25: continue
        # Laplace shrinkage toward unconditional past event frequency
        base=np.mean([outcome[i] for i,_ in eligible])
        p=(sum(matches)+20*base)/(len(matches)+20)
        predictions.append((t,float(p),outcome[t],float(base)))
    current=features[-1] if features else None
    if current is None: return {"status":"insufficient_history"}
    t,bucket,momentum=current
    eligible=[(i,b) for i,b,_ in features if i+horizon<t and i in outcome]
    matches=[outcome[i] for i,b in eligible if b==bucket]
    if len(eligible)<120 or len(matches)<25:
        return {"status":"insufficient_history","regime":bucket,"matching_samples":len(matches)}
    base=float(np.mean([outcome[i] for i,_ in eligible]))
    prob=float((sum(matches)+20*base)/(len(matches)+20))
    result={"status":"experimental","event":"5d_close_drawdown","threshold_pct":round(threshold*100,1),
      "horizon_days":horizon,"regime":bucket,"probability_pct":round(100*prob,1),
      "base_rate_pct":round(100*base,1),"matching_samples":len(matches),
      "training_samples":len(eligible),"momentum20_pct":round(100*momentum,2),
      "signal":("risk_high" if prob>=max(0.35,base*1.5) else "risk_elevated" if prob>=max(0.2,base*1.2) else "risk_normal")}
    if len(predictions)>=30:
        brier=np.mean([(p-y)**2 for _,p,y,_ in predictions])
        baseline=np.mean([(b-y)**2 for _,p,y,b in predictions])
        result["walk_forward"]={"samples":len(predictions),"brier":round(float(brier),4),
            "baseline_brier":round(float(baseline),4),"beats_baseline":bool(brier<baseline),"validated":bool(brier<baseline and len(predictions)>=100),"evaluation":"expanding_past_only_with_overlapping_5d_labels"}
    result["signal_eligible"]=bool(result.get("walk_forward",{}).get("validated",False))
    if not result["signal_eligible"]: result["signal"]="disabled_unvalidated"
    return result

def calc(symbol):
    frame = yf.download(symbol, period="2y", interval="1d", auto_adjust=False, progress=False, threads=False)
    if frame.empty: raise RuntimeError("Empty history for "+symbol)
    close = frame["Close"]
    if isinstance(close,pd.DataFrame): close=close.iloc[:,0]
    close = pd.to_numeric(close,errors="coerce").dropna()
    if len(close)<100: raise RuntimeError("Insufficient observations for "+symbol)
    if not np.isfinite(close.to_numpy()).all() or (close<=0).any(): raise RuntimeError("Invalid/non-positive prices for "+symbol)
    if not close.index.is_monotonic_increasing or close.index.has_duplicates: raise RuntimeError("Invalid dates for "+symbol)
    r=np.log(close).diff().dropna()
    if not np.isfinite(r.to_numpy()).all(): raise RuntimeError("Invalid returns for "+symbol)
    vol20=float(r.tail(20).std()*math.sqrt(252)*100)
    vol60=float(r.tail(60).std()*math.sqrt(252)*100)
    rolling=r.rolling(20).std()*math.sqrt(252)*100
    valid=rolling.dropna()
    if len(valid)<60: raise RuntimeError("Insufficient volatility history for "+symbol)
    percentile=float((valid<=vol20).mean()*100)
    peak=float(close.cummax().iloc[-1])
    drawdown=(float(close.iloc[-1])/peak-1)*100
    dates=[x.strftime("%Y-%m-%d") for x in close.index]
    return {"symbol":symbol,"asof":dates[-1],"close":round(float(close.iloc[-1]),2),
            "vol20_pct":round(vol20,2),"vol60_pct":round(vol60,2),
            "vol_percentile_2y":round(percentile,1),"drawdown_pct":round(drawdown,2),
            "regime":("high" if percentile>=80 else "elevated" if percentile>=60 else "normal"),
            "observations":len(close),"risk_model":risk_model(close,symbol),
            "history":[{"date":d,"close":round(float(v),2)} for d,v in zip(dates,close.values)]}
def main():
    result={"generated_at":datetime.now(timezone.utc).isoformat(),"source":"Yahoo Finance via yfinance","method":"20d annualized realized volatility; percentile within available 2y sample; experimental 5d downside-risk conditional frequency with expanding walk-forward validation; not a trade forecast","markets":{},"errors":{}}
    for name,ticker in {"ES":"ES=F","NQ":"NQ=F"}.items():
        try: result["markets"][name]=calc(ticker)
        except Exception as exc: result["errors"][name]=str(exc)
    if len(result["markets"])!=2: raise RuntimeError("Incomplete ES/NQ data; refusing to overwrite last complete snapshot: "+str(result["errors"]))
    with open("data.json","w",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,allow_nan=False,separators=(",",":"))
    print("Snapshot generated:",list(result["markets"]),"errors:",result["errors"])
if __name__=="__main__":main()
