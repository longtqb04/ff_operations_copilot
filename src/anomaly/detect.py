import numpy as np
import pandas as pd
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.config import ANOMALY_PATH, ANOMALY_METRICS_PATH, ANOMALY_RESIDUAL_THRESHOLD, FORECAST_PATH, MIN_REVENUE_GAP

def detect_anomalies(source=FORECAST_PATH):
    raw=pd.read_csv(source,low_memory=False); raw["date"]=pd.to_datetime(raw.date)
    for col in ("event_timestamp","incident_started_at","data_available_at"):
        raw[col]=pd.to_datetime(raw[col],errors="coerce")
    groups=["store_id","date","daypart","channel","region","store_type"]
    df=raw.groupby(groups,as_index=False).agg(actual_sales=("actual_sales","sum"),forecast_sales=("forecast_sales","sum"),transaction_count=("transaction_count","sum"),quantity=("quantity","sum"),lag_7=("lag_7","sum"),delivery_eta=("delivery_eta","mean"),stockout_flag=("stockout_flag","max"),promotion_flag=("promotion_flag","max"),is_actionable_anomaly=("is_actionable_anomaly","max"),event_timestamp=("event_timestamp","min"),incident_started_at=("incident_started_at","min"),data_available_at=("data_available_at","max"))
    df["average_order_value"]=df.actual_sales/df.transaction_count.clip(lower=1)
    df["residual"]=df.actual_sales-df.forecast_sales; df["residual_pct"]=df.residual/df.forecast_sales.clip(lower=1); df["absolute_residual"]=df.residual.abs()
    df=df.sort_values(["store_id","daypart","channel","date"])
    df["sales_change_vs_last_week"] = df.groupby(["store_id", "daypart", "channel"], observed=True).actual_sales.pct_change(7).replace([np.inf, -np.inf], 0).fillna(0)

    x = StandardScaler().fit_transform(df[["residual_pct", "absolute_residual", "sales_change_vs_last_week"]].fillna(0))
    detector = IsolationForest(contamination=.025, random_state=42)
    pred = detector.fit_predict(x)
    score = -detector.score_samples(x)

    df["anomaly_score"] = (score - score.min()) / (score.max() - score.min() + 1e-9)
    df["isolation_flag"] = pred == -1
    df["is_anomaly"] = df.isolation_flag & (df.residual_pct.abs() >= ANOMALY_RESIDUAL_THRESHOLD) & (df.absolute_residual >= MIN_REVENUE_GAP)
    df["severity"] = np.select([df.is_anomaly&(df.residual_pct.abs()>=.35),df.is_anomaly&(df.residual_pct.abs()>=.25),df.is_anomaly],["critical","high","medium"],default="normal")
    df["detected_at"]=df.data_available_at+pd.Timedelta(minutes=20)
    df["detection_lag_minutes"]=(df.detected_at-df.incident_started_at).dt.total_seconds()/60

    labels=df.is_actionable_anomaly.astype(bool); flags=df.is_anomaly.astype(bool)
    tp=int((flags&labels).sum()); fp=int((flags&~labels).sum()); fn=int((~flags&labels).sum())
    lags=df.loc[flags&labels,"detection_lag_minutes"].dropna()
    metrics={"true_positives":tp,"false_positives":fp,"false_negatives":fn,"precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1),"mean_detection_lag_minutes":float(lags.mean()) if len(lags) else None,"p95_detection_lag_minutes":float(lags.quantile(.95)) if len(lags) else None,"target_precision_met":tp/max(tp+fp,1)>=.8,"target_detection_lag_met":bool(len(lags) and lags.quantile(.95)<120),"evaluation_mode":"synthetic_ground_truth"}
    
    result = df[df.is_anomaly].sort_values(["anomaly_score","absolute_residual"],ascending=False)
    ANOMALY_PATH.parent.mkdir(parents=True,exist_ok=True); result.to_csv(ANOMALY_PATH,index=False); ANOMALY_METRICS_PATH.write_text(json.dumps(metrics,indent=2),encoding="utf-8"); return result
