"""Public-data MTM snapshot. No brokerage credentials or Credit Pulse integration."""
import json, math, os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

def calc(symbol):
    frame = yf.download(symbol, period="2y", interval="1d", auto_adjust=False, progress=False, threads=False)
    if frame.empty: raise RuntimeError("Empty history for "+symbol)
    close = frame["Close"]
    if isinstance(close,pd.DataFrame): close=close.iloc[:,0]
    close = pd.to_numeric(close,errors="coerce").dropna()
    if len(close)<100: raise RuntimeError("Insufficient observations for "+symbol)
    r=np.log(close).diff().dropna()
    vol20=float(r.tail(20).std()*math.sqrt(252)*100)
    vol60=float(r.tail(60).std()*math.sqrt(252)*100)
    rolling=r.rolling(20).std()*math.sqrt(252)*100
    valid=rolling.dropna()
    percentile=float((valid<=vol20).mean()*100)
    peak=float(close.cummax().iloc[-1])
    drawdown=(float(close.iloc[-1])/peak-1)*100
    dates=[x.strftime("%Y-%m-%d") for x in close.index]
    return {"symbol":symbol,"asof":dates[-1],"close":round(float(close.iloc[-1]),2),
            "vol20_pct":round(vol20,2),"vol60_pct":round(vol60,2),
            "vol_percentile_2y":round(percentile,1),"drawdown_pct":round(drawdown,2),
            "regime":("high" if percentile>=80 else "elevated" if percentile>=60 else "normal"),
            "observations":len(close),
            "history":[{"date":d,"close":round(float(v),2)} for d,v in zip(dates,close.values)]}
def main():
    result={"generated_at":datetime.now(timezone.utc).isoformat(),"source":"Yahoo Finance via yfinance","method":"20d annualized realized volatility; percentile within available 2y sample; no forecast","markets":{},"errors":{}}
    for name,ticker in {"ES":"ES=F","NQ":"NQ=F"}.items():
        try: result["markets"][name]=calc(ticker)
        except Exception as exc: result["errors"][name]=str(exc)
    if not result["markets"]: raise RuntimeError("No valid market data; refusing to overwrite last valid snapshot")
    with open("data.json","w",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,allow_nan=False,separators=(",",":"))
    print("Snapshot generated:",list(result["markets"]),"errors:",result["errors"])
if __name__=="__main__":main()
