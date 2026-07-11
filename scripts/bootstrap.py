from src.anomaly.detect import detect_anomalies
from src.data.generate import generate_sales
from src.forecasting.pipeline import train_and_predict

if __name__ == "__main__":
    sales = generate_sales()
    metrics = train_and_predict()
    anomalies = detect_anomalies()
    print(f"Rows: {len(sales):,}"); print("Metrics:",{k:round(v,4) for k,v in metrics.items()}); print(f"Anomalies: {len(anomalies)}")

