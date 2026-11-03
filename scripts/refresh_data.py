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
    if symbol not in ("ES=F","NQ=F"): raise ValueError("Unsupported futures symbol: "+str(symbol))
    threshold = 0.02 if symbol == "ES=F" else 0.03
    horizon = 5
    values = close.to_numpy(dtype=float)
    if len(values)<250 or not np.isfinite(values).all() or np.any(values<=0):
        return {"status":"insufficient_history","signal":"disabled_unvalidated","signal_eligible":False,"threshold_pct":round(threshold*100,1)}
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
    if current is None: return {"status":"insufficient_history","signal":"disabled_unvalidated","signal_eligible":False,"threshold_pct":round(threshold*100,1)}
    t,bucket,momentum=current
    eligible=[(i,b) for i,b,_ in features if i+horizon<t and i in outcome]
    matches=[outcome[i] for i,b in eligible if b==bucket]
    if len(eligible)<120 or len(matches)<25:
        return {"status":"insufficient_history","regime":bucket,"matching_samples":len(matches),"signal":"disabled_unvalidated","signal_eligible":False,"threshold_pct":round(threshold*100,1)}
    base=float(np.mean([outcome[i] for i,_ in eligible]))
    prob=float((sum(matches)+20*base)/(len(matches)+20))
    result={"status":"experimental","event":"5d_close_drawdown","threshold_pct":round(threshold*100,1),
      "horizon_days":horizon,"regime":bucket,"probability_pct":round(100*prob,1),
      "base_rate_pct":round(100*base,1),"matching_samples":len(matches),
      "training_samples":len(eligible),"momentum20_pct":round(100*momentum,2),
      "signal":("risk_high" if prob>=max(0.35,base*1.5) else "risk_elevated" if prob>=max(0.2,base*1.2) else "risk_normal")}
    # Non-overlapping evaluation anchors: only one 5-day label per block.
    # Train strictly on outcomes whose full 5-day horizon has ended.
    independent = [x for x in predictions if x[0] % horizon == 0]
    # Keep all five-day test labels disjoint. Training labels must finish before each test anchor.
    # Causal, regularized additive calibration: use only prior, fully matured,
    # nonoverlapping out-of-sample predictions. Never fit to the test label.
    def calibrate(anchor, raw):
        prior = [x for x in independent if x[0] + horizon < anchor]
        if len(prior) < 40:
            return float(raw), 0, 0.0
        bias = float(np.mean([y - p for _, p, y, _ in prior]))
        correction = float(np.clip(bias * len(prior) / (len(prior) + 40), -0.15, 0.15))
        return float(np.clip(raw + correction, 0.001, 0.999)), len(prior), correction

    calibrated_tests = []
    for anchor, raw, observed, baseline_p in independent:
        adjusted, count, correction = calibrate(anchor, raw)
        if count >= 40:
            calibrated_tests.append((anchor, raw, adjusted, observed, baseline_p))
    current_adjusted, calibration_samples, calibration_correction = calibrate(t, prob)
    result["raw_probability_pct"] = round(100 * prob, 1)
    result["calibrated_probability_pct"] = round(100 * current_adjusted, 1)
    result["probability_pct"] = round(100 * current_adjusted, 1)
    result["calibration"] = {
        "method":"past_only_regularized_additive_bias",
        "training_nonoverlapping_predictions":calibration_samples,
        "correction_percentage_points":round(100 * calibration_correction, 2),
        "minimum_samples":40,
        "prior_shrinkage":40,
        "max_absolute_correction_pp":15,
        "validated":False
    }
    if calibrated_tests:
        adjusted_brier = float(np.mean([(a-y)**2 for _,r,a,y,b in calibrated_tests]))
        raw_comparable_brier = float(np.mean([(r-y)**2 for _,r,a,y,b in calibrated_tests]))
        comparable_baseline = float(np.mean([(b-y)**2 for _,r,a,y,b in calibrated_tests]))
        calibration_bias = float(np.mean([a-y for _,r,a,y,b in calibrated_tests]))
        # Chronological last-20% diagnostic, evaluated after all calibration rules are fixed.
        holdout = calibrated_tests[int(len(calibrated_tests)*0.8):]
        if len(holdout) >= 20:
            # Row format: (anchor, raw, adjusted, outcome, baseline).
            calibrated_holdout = float(np.mean([(a-y)**2 for _,r,a,y,b in holdout]))
            raw_holdout = float(np.mean([(r-y)**2 for _,r,a,y,b in holdout]))
            baseline_holdout = float(np.mean([(b-y)**2 for _,r,a,y,b in holdout]))
            result["calibration"]["chronological_holdout"] = {
                "samples":len(holdout),
                "raw_brier":round(raw_holdout,5),
                "calibrated_brier":round(calibrated_holdout,5),
                "baseline_brier":round(baseline_holdout,5),
                "beats_baseline":bool(calibrated_holdout < baseline_holdout),
                "pristine_holdout":False,
                "limitation":"Historical data previously examined during development; prospective confirmation required"
            }
        result["calibration"]["out_of_sample"] = {
            "samples":len(calibrated_tests),
            "calibrated_brier":round(adjusted_brier,5),
            "raw_brier_same_period":round(raw_comparable_brier,5),
            "baseline_brier_same_period":round(comparable_baseline,5),
            "mean_forecast_minus_event_rate_pp":round(100*calibration_bias,2),
            "improves_raw":bool(adjusted_brier < raw_comparable_brier),
            "beats_baseline":bool(adjusted_brier < comparable_baseline)
        }

    if len(independent) >= 20:
        brier = float(np.mean([(p-y)**2 for _,p,y,_ in independent]))
        baseline = float(np.mean([(b-y)**2 for _,p,y,b in independent]))
        wins = brier < baseline
        actual_rate = float(np.mean([y for _,p,y,b in independent]))
        forecast_mean = float(np.mean([p for _,p,y,b in independent]))
        skill = 1.0 - brier/baseline if baseline > 0 else None
        # Paired bootstrap over nonoverlapping 5-session test windows.
        losses = np.asarray([(b-y)**2-(p-y)**2 for _,p,y,b in independent], dtype=float)
        rng = np.random.default_rng(20261103)
        draws = rng.integers(0,len(losses),size=(1000,len(losses)))
        improvements = losses[draws].mean(axis=1)
        ci_low,ci_high = np.percentile(improvements,[2.5,97.5])
        result["walk_forward"] = {
            "samples":len(independent), "brier":round(brier,4),
            "actual_event_rate_pct":round(100*actual_rate,1),
            "mean_forecast_pct":round(100*forecast_mean,1),
            "brier_skill_score":round(float(skill),4) if skill is not None else None,
            "brier_improvement_ci95":[round(float(ci_low),5),round(float(ci_high),5)],
            "statistically_positive":bool(ci_low>0),
            "confidence_method":"paired_iid_bootstrap_nonoverlapping_5session_windows_1000_reps",
            "baseline_brier":round(baseline,4), "beats_baseline":bool(wins),
            "validated":False,
            "evaluation":"nonoverlapping_5d_test_labels; expanding_train; embargo_5d",
            "limitations":"Unverified continuous futures roll adjustments; calibration and trading costs not independently validated"}
    else:
        result["walk_forward"] = {"samples":len(independent),"validated":False,
          "evaluation":"nonoverlapping_5d_test_labels","limitations":"Insufficient independent tests"}
    # Do not turn on trade signals merely because one backtest beats baseline.
    # Require independent contract-roll validation, longer data, calibration,
    # and transaction-cost tests before any live trading recommendation.
    result["signal_eligible"]=False
    result["signal"]="disabled_unvalidated"
    return result

def calc(symbol):
    frame = yf.download(symbol, period="5y", interval="1d", auto_adjust=False, progress=False, threads=False)
    if frame.empty: raise RuntimeError("Empty history for "+symbol)
    close = frame["Close"]
    if isinstance(close,pd.DataFrame): close=close.iloc[:,0]
    close = pd.to_numeric(close,errors="coerce").dropna()
    if len(close)<250: raise RuntimeError("Insufficient observations for "+symbol)
    if not np.isfinite(close.to_numpy()).all() or (close<=0).any(): raise RuntimeError("Invalid/non-positive prices for "+symbol)
    if not close.index.is_monotonic_increasing or close.index.has_duplicates: raise RuntimeError("Invalid dates for "+symbol)
    r=np.log(close).diff().dropna()
    # Futures continuation can contain contract-roll discontinuities. Flag them; do not hide them.
    large_jumps=r[np.abs(r)>0.08]
    roll_warning=bool(len(large_jumps)>0)
    jump_dates=[x.strftime("%Y-%m-%d") for x in large_jumps.index]
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
            "data_quality":{"large_daily_log_returns_over_8pct":len(large_jumps),
              "suspect_dates":jump_dates[:20],"roll_adjustment_verified":False,
              "warning":"Continuous futures contract rolls are not verified" if not roll_warning else "Large daily jumps detected; possible contract roll or genuine volatility"},
            "history":[{"date":d,"close":round(float(v),2)} for d,v in zip(dates,close.values)]}
def main():
    result={"generated_at":datetime.now(timezone.utc).isoformat(),"source":"Yahoo Finance via yfinance; ES=F and NQ=F continuous front-month proxies; roll adjustments unverified","method":"20d realized volatility; percentile within available sample (up to 5y); experimental 5d downside-risk model with nonoverlapping walk-forward validation; trading signals disabled","markets":{},"errors":{}}
    for name,ticker in {"ES":"ES=F","NQ":"NQ=F"}.items():
        try: result["markets"][name]=calc(ticker)
        except Exception as exc: result["errors"][name]=str(exc)
    if len(result["markets"])!=2: raise RuntimeError("Incomplete ES/NQ data; refusing to overwrite last complete snapshot: "+str(result["errors"]))
    with open("data.json","w",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,allow_nan=False,separators=(",",":"))
    print("Snapshot generated:",list(result["markets"]),"errors:",result["errors"])
if __name__=="__main__":main()
