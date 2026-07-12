from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
SALES_PATH = DATA_DIR / "sales.csv"
FORECAST_PATH = DATA_DIR / "forecasts.csv"
ANOMALY_PATH = DATA_DIR / "anomalies.csv"
ANOMALY_METRICS_PATH = DATA_DIR / "anomaly_metrics.json"
EXTERNAL_SIGNALS_PATH = DATA_DIR / "external_signals.json"
MODEL_PATH = MODEL_DIR / "forecast.joblib"

RANDOM_SEED = 42
FORECAST_HORIZON_DAYS = 30
ANOMALY_RESIDUAL_THRESHOLD = 0.15
MIN_REVENUE_GAP = 5_000_000
