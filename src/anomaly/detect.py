import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.config import ANOMALY_PATH, ANOMALY_RESIDUAL_THRESHOLD, FORECAST_PATH, MIN_REVENUE_GAP

def detect_anomalies(source=FORECAST_PATH):
    df = pd.read_csv(source)

    df["date"] = pd.to_datetime(df.date)
    df["sales_change_vs_last_week"] = df.groupby(["store_id", "daypart", "channel"], observed=True).actual_sales.pct_change(7).replace([np.inf, -np.inf], 0).fillna(0)

    x = StandardScaler().fit_transform(df[["residual_pct", "absolute_residual", "sales_change_vs_last_week"]].fillna(0))
    detector = IsolationForest(contamination=.025, random_state=42)
    pred = detector.fit_predict(x)
    score = -detector.score_samples(x)

    df["anomaly_score"] = (score - score.min()) / (score.max() - score.min() + 1e-9)
    df["isolation_flag"] = pred == -1
    df["is_anomaly"] = df.isolation_flag & (df.residual_pct.abs() >= ANOMALY_RESIDUAL_THRESHOLD) & (df.absolute_residual >= MIN_REVENUE_GAP)
    df["severity"] = np.select([df.is_anomaly&(df.residual_pct.abs()>=.35),df.is_anomaly&(df.residual_pct.abs()>=.25),df.is_anomaly],["critical","high","medium"],default="normal")
    
    result = df[df.is_anomaly].sort_values(["anomaly_score","absolute_residual"],ascending=False)
    ANOMALY_PATH.parent.mkdir(parents=True,exist_ok=True); result.to_csv(ANOMALY_PATH,index=False); return result

