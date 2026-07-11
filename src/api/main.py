import json
import secrets
import threading
import os
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
load_dotenv()
from src.anomaly.detect import detect_anomalies
from src.config import ANOMALY_PATH, FORECAST_PATH, MODEL_PATH, SALES_PATH
from src.copilot.investigate import investigate, simulate
from src.data.generate import generate_sales
from src.forecasting.pipeline import train_and_predict
from src.agent.workflow import run_investigation_agent
from src.external.apify import external_context,ingest_webhook,run_configured_actor,status as apify_status

def ensure_artifacts():
    if not SALES_PATH.exists(): generate_sales()
    if not MODEL_PATH.exists() or not FORECAST_PATH.exists(): train_and_predict()
    if not ANOMALY_PATH.exists(): detect_anomalies()

@asynccontextmanager
async def lifespan(_):
    ensure_artifacts(); yield

app=FastAPI(title="Fast Food Operations Copilot",version="0.1.0",lifespan=lifespan)
refresh_lock=threading.Lock()
DASHBOARD_PATH = Path(__file__).with_name("dashboard.html")
STORE_MONITOR_PATH = Path(__file__).with_name("store_monitoring.html")
INVESTIGATIONS_PATH = Path(__file__).with_name("investigations.html")
WHAT_IF_PATH = Path(__file__).with_name("what_if.html")

@app.get("/",response_class=HTMLResponse)
def home():
    return DASHBOARD_PATH.read_text(encoding="utf-8")

@app.get("/store-monitoring",response_class=HTMLResponse)
def store_monitoring_page():
    return STORE_MONITOR_PATH.read_text(encoding="utf-8")

@app.get("/investigations",response_class=HTMLResponse)
def investigations_page():
    return INVESTIGATIONS_PATH.read_text(encoding="utf-8")

@app.get("/what-if",response_class=HTMLResponse)
def what_if_page():
    return WHAT_IF_PATH.read_text(encoding="utf-8")

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/api/agent/status")
def agent_status():
    return {"agent_enabled":True,"llm_configured":bool(os.getenv("OPENAI_API_KEY")),"mode":"llm_tool_calling" if os.getenv("OPENAI_API_KEY") else "deterministic_fallback","model":os.getenv("OPENAI_AGENT_MODEL","gpt-5.4-mini")}

@app.get("/api/external/apify/status")
def get_apify_status(): return apify_status()

@app.get("/api/external/apify/signals/{store_id}")
def get_apify_signals(store_id:str): return external_context(store_id.upper())

@app.post("/api/external/apify/refresh")
def refresh_apify_signals():
    try:
        result=run_configured_actor(); return {"status":"refreshed","mode":result["mode"],"signal_count":len(result["items"])}
    except Exception as exc: raise HTTPException(status_code=502,detail=f"Apify refresh failed: {exc}") from exc

@app.post("/api/external/apify/webhook")
def receive_apify_webhook(payload:dict,x_apify_webhook_secret:str|None=Header(default=None)):
    expected=os.getenv("APIFY_WEBHOOK_SECRET")
    if not expected or x_apify_webhook_secret!=expected: raise HTTPException(status_code=401,detail="Invalid or unconfigured webhook secret")
    try: items=ingest_webhook(payload); return {"status":"accepted","signal_count":len(items)}
    except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc

@app.post("/api/agent/investigate/{incident_id}")
def run_agent_investigation(incident_id:str):
    try: return run_investigation_agent(incident_id)
    except ValueError as exc: raise HTTPException(status_code=404,detail=str(exc)) from exc
    except Exception as exc: raise HTTPException(status_code=502,detail=f"Agent execution failed: {exc}") from exc

@app.post("/api/refresh")
def refresh_data():
    if not refresh_lock.acquire(blocking=False):
        raise HTTPException(status_code=409,detail="A data refresh is already running")
    try:
        seed=secrets.randbelow(2_000_000_000)+1
        sales=generate_sales(seed=seed,shuffle=True)
        metrics=train_and_predict()
        anomalies=detect_anomalies()
        return {"status":"refreshed","seed":seed,"rows":len(sales),"anomalies":len(anomalies),"model_metrics":metrics}
    finally:
        refresh_lock.release()

@app.get("/api/overview")
def overview():
    df=pd.read_csv(FORECAST_PATH); anomalies=pd.read_csv(ANOMALY_PATH); metrics=joblib.load(MODEL_PATH)["metrics"]
    return {"actual_sales":int(df.actual_sales.sum()),"forecast_sales":int(df.forecast_sales.sum()),"revenue_gap":int((df.actual_sales-df.forecast_sales).sum()),"anomaly_count":len(anomalies),"stores_at_risk":int(anomalies.store_id.nunique()) if len(anomalies) else 0,"model_metrics":metrics}

@app.get("/api/anomalies")
def anomalies(limit:int=Query(20,ge=1,le=100)):
    return json.loads(pd.read_csv(ANOMALY_PATH).head(limit).to_json(orient="records",date_format="iso"))

@app.get("/api/trend")
def trend(days:int=Query(14,ge=7,le=30)):
    df=pd.read_csv(FORECAST_PATH); df["date"]=pd.to_datetime(df.date)
    daily=df.groupby("date",as_index=False)[["actual_sales","forecast_sales"]].sum().tail(days)
    daily["revenue_gap"]=daily.actual_sales-daily.forecast_sales
    return json.loads(daily.to_json(orient="records",date_format="iso"))

@app.get("/api/store-performance")
def store_performance(limit:int=Query(8,ge=3,le=20)):
    df=pd.read_csv(FORECAST_PATH)
    stores=df.groupby(["store_id","region"],as_index=False)[["actual_sales","forecast_sales"]].sum()
    stores["revenue_gap"]=stores.actual_sales-stores.forecast_sales
    stores["variance_pct"]=stores.revenue_gap/stores.forecast_sales.clip(lower=1)
    stores=stores.sort_values("variance_pct").head(limit)
    return json.loads(stores.to_json(orient="records"))

@app.get("/api/stores")
def stores():
    df=pd.read_csv(FORECAST_PATH)
    result=df[["store_id","region","store_type"]].drop_duplicates().sort_values("store_id")
    return json.loads(result.to_json(orient="records"))

@app.get("/api/store-monitor/{store_id}")
def store_monitor(store_id:str,days:int=Query(14,ge=7,le=30)):
    df=pd.read_csv(FORECAST_PATH); df["date"]=pd.to_datetime(df.date); store_id=store_id.upper()
    store=df[df.store_id==store_id].copy()
    if store.empty: raise HTTPException(status_code=404,detail="Store not found")
    max_date=store.date.max(); recent=store[store.date>max_date-pd.Timedelta(days=days)]
    actual=float(recent.actual_sales.sum()); forecast=float(recent.forecast_sales.sum()); gap=actual-forecast
    timeline=recent.groupby("date",as_index=False)[["actual_sales","forecast_sales"]].sum(); timeline["revenue_gap"]=timeline.actual_sales-timeline.forecast_sales
    channels=recent.groupby("channel",as_index=False)[["actual_sales","forecast_sales","transaction_count"]].sum(); channels["revenue_gap"]=channels.actual_sales-channels.forecast_sales; channels["variance_pct"]=channels.revenue_gap/channels.forecast_sales.clip(lower=1)
    dayparts=recent.groupby(["daypart","channel"],as_index=False)[["actual_sales","forecast_sales"]].sum(); dayparts["variance_pct"]=(dayparts.actual_sales-dayparts.forecast_sales)/dayparts.forecast_sales.clip(lower=1)
    all_recent=df[df.date>max_date-pd.Timedelta(days=days)].copy(); summary=all_recent.groupby(["store_id","region"],as_index=False)[["actual_sales","forecast_sales"]].sum(); summary["variance_pct"]=(summary.actual_sales-summary.forecast_sales)/summary.forecast_sales.clip(lower=1)
    region=store.region.iloc[0]; peer=summary[(summary.region==region)&(summary.store_id!=store_id)].variance_pct.median(); network=summary.variance_pct.median()
    anomalies=pd.read_csv(ANOMALY_PATH); incidents=anomalies[anomalies.store_id==store_id].copy()
    tx=int(recent.transaction_count.sum()); aov=int(actual/max(tx,1)); eta=float(recent.delivery_eta.mean()); stockouts=int(recent.stockout_flag.sum())
    payload={"store":{"store_id":store_id,"region":region,"store_type":store.store_type.iloc[0],"status":"critical" if len(incidents) else ("watch" if gap/forecast<-.05 else "healthy"),"as_of":str(max_date.date())},
      "kpis":{"actual_sales":int(actual),"forecast_sales":int(forecast),"revenue_gap":int(gap),"variance_pct":float(gap/max(forecast,1)),"transaction_count":tx,"average_order_value":aov,"anomaly_count":len(incidents),"estimated_recoverable_revenue":int(max(0,-gap)*.45)},
      "benchmark":{"store":float(gap/max(forecast,1)),"peer_median":float(peer),"region":float(summary[summary.region==region].variance_pct.median()),"network":float(network)},
      "signals":[{"name":"Delivery ETA","value":round(eta,1),"unit":"min","status":"critical" if eta>35 else "normal"},{"name":"Stock-out events","value":stockouts,"unit":"events","status":"warning" if stockouts else "normal"},{"name":"Promotion coverage","value":round(float(recent.promotion_flag.mean()*100),1),"unit":"%","status":"context"},{"name":"Average rainfall","value":round(float(recent.rainfall.mean()),1),"unit":"mm","status":"context"}],
      "timeline":json.loads(timeline.to_json(orient="records",date_format="iso")),"channels":json.loads(channels.to_json(orient="records")),"dayparts":json.loads(dayparts.to_json(orient="records")),"ranking":json.loads(summary.sort_values("variance_pct").to_json(orient="records")),"incidents":json.loads(incidents.head(10).to_json(orient="records",date_format="iso"))}
    return payload

@app.get("/api/investigations")
def investigation_queue():
    df=pd.read_csv(ANOMALY_PATH); df["date"]=pd.to_datetime(df.date)
    if df.empty: return []
    df["incident_id"]=df.apply(lambda r:f"{r.store_id}-{r.date:%Y%m%d}-{r.daypart}-{r.channel}",axis=1)
    df["status"]="new"; df["priority_score"]=(df.anomaly_score*df.absolute_residual).round()
    cols=["incident_id","store_id","date","region","daypart","channel","severity","status","actual_sales","forecast_sales","residual","residual_pct","anomaly_score","priority_score"]
    return json.loads(df.sort_values("priority_score",ascending=False)[cols].to_json(orient="records",date_format="iso"))

@app.get("/api/investigations/{incident_id}")
def investigation_detail(incident_id:str):
    anomalies=pd.read_csv(ANOMALY_PATH); anomalies["date"]=pd.to_datetime(anomalies.date)
    anomalies["incident_id"]=anomalies.apply(lambda r:f"{r.store_id}-{r.date:%Y%m%d}-{r.daypart}-{r.channel}",axis=1)
    match=anomalies[anomalies.incident_id==incident_id]
    if match.empty: raise HTTPException(status_code=404,detail="Investigation not found")
    event=match.iloc[0]; date=str(event.date.date()); report=investigate(event.store_id,date)
    df=pd.read_csv(FORECAST_PATH); df["date"]=pd.to_datetime(df.date)
    series=df[(df.store_id==event.store_id)&(df.daypart==event.daypart)&(df.channel==event.channel)&(df.date.between(event.date-pd.Timedelta(days=6),event.date+pd.Timedelta(days=2)))][["date","actual_sales","forecast_sales","delivery_eta","stockout_flag"]]
    peer=df[(df.date==event.date)&(df.region==event.region)&(df.store_id!=event.store_id)&(df.daypart==event.daypart)&(df.channel==event.channel)]
    peer_variance=float(((peer.actual_sales-peer.forecast_sales)/peer.forecast_sales.clip(lower=1)).median()) if len(peer) else 0
    baseline_tx=event.lag_7/max(event.average_order_value,1); tx_change=float(event.transaction_count/max(baseline_tx,1)-1); eta_delta=float(event.delivery_eta-27)
    evidence=[{"label":"Delivery transactions","current":int(event.transaction_count),"baseline":round(baseline_tx),"change_pct":round(tx_change,3),"source":"POS transactions","confidence":.94,"signal":"critical"},{"label":"Delivery ETA","current":round(float(event.delivery_eta),1),"baseline":27,"unit":"minutes","change":round(eta_delta,1),"source":"Delivery operations","confidence":.88,"signal":"critical" if eta_delta>8 else "normal"},{"label":"Peer performance","current":round(peer_variance,3),"baseline":0,"unit":"variance","source":"Peer store cluster","confidence":.91,"signal":"normal"},{"label":"Product availability","current":int(event.stockout_flag),"baseline":0,"unit":"events","source":"Inventory signal","confidence":.76,"signal":"warning" if event.stockout_flag else "normal"}]
    hypotheses=[{"title":"Delivery platform or integration disruption","confidence":.87,"impact":int(event.absolute_residual*.62),"support":["Delivery revenue materially below forecast","Peer stores remained stable","Issue isolated to delivery channel"],"contradiction":"Platform uptime confirmation not yet available"},{"title":"Kitchen throughput increased delivery ETA","confidence":.82,"impact":int(event.absolute_residual*.25),"support":[f"Delivery ETA increased to {event.delivery_eta:.0f} minutes","Lunch is the affected daypart"],"contradiction":"Direct kitchen telemetry is unavailable"},{"title":"Top-product stock-out reduced conversion","confidence":.76 if event.stockout_flag else .42,"impact":int(event.absolute_residual*.13),"support":["Inventory signal detected" if event.stockout_flag else "No confirmed stock-out on this record"],"contradiction":"Category-level sales are not yet connected"}]
    actions=[]
    for i,a in enumerate(report["recommended_actions"]): actions.append({**a,"expected_impact_low":int(event.absolute_residual*(.28-.04*i)),"expected_impact_high":int(event.absolute_residual*(.48-.05*i)),"effort":["low","medium","low"][min(i,2)],"owner":["Store manager","Operations lead","Inventory lead"][min(i,2)]})
    return {"incident":{"incident_id":incident_id,"store_id":event.store_id,"date":date,"region":event.region,"daypart":event.daypart,"channel":event.channel,"severity":event.severity,"status":"new","actual_sales":int(event.actual_sales),"forecast_sales":int(event.forecast_sales),"revenue_gap":int(event.residual),"residual_pct":float(event.residual_pct),"anomaly_score":float(event.anomaly_score)},"summary":report["executive_summary"],"scope_assessment":report["scope_assessment"],"peer_variance":peer_variance,"timeline":json.loads(series.to_json(orient="records",date_format="iso")),"decomposition":report["drivers"],"evidence":evidence,"hypotheses":hypotheses,"actions":actions,"disclaimer":report["disclaimer"]}

@app.get("/api/scenario/{incident_id}")
def run_scenario(incident_id:str,delivery_recovery:float=Query(.5,ge=0,le=1),promotion_uplift:float=Query(0,ge=0,le=.5),target_eta:float=Query(27,ge=15,le=60),stockout_recovery:float=Query(0,ge=0,le=1)):
    anomalies=pd.read_csv(ANOMALY_PATH); anomalies["date"]=pd.to_datetime(anomalies.date)
    anomalies["incident_id"]=anomalies.apply(lambda r:f"{r.store_id}-{r.date:%Y%m%d}-{r.daypart}-{r.channel}",axis=1)
    match=anomalies[anomalies.incident_id==incident_id]
    if match.empty: raise HTTPException(status_code=404,detail="Incident not found")
    event=match.iloc[0]; actual=float(event.actual_sales); baseline=float(event.forecast_sales); gap=max(0,baseline-actual)
    def calculate(recovery):
        delivery_gain=gap*recovery*.72
        eta_room=max(0,float(event.delivery_eta)-27); eta_gain=gap*.16*(max(0,float(event.delivery_eta)-target_eta)/max(eta_room,1))
        stock_gain=gap*.12*stockout_recovery if event.stockout_flag else 0
        promo_gain=baseline*promotion_uplift
        uplift=max(0,delivery_gain+eta_gain+stock_gain+promo_gain)
        return min(uplift,baseline*1.5-actual)
    uplift=calculate(delivery_recovery); scenario=max(0,actual+uplift); remaining=baseline-scenario
    distance=delivery_recovery*.08+promotion_uplift*.35+abs(target_eta-27)/100+stockout_recovery*.04
    confidence=max(.48,min(.9,.9-distance)); uncertainty=uplift*(1-confidence)*.65
    levels=[0,.25,.5,.75,1]
    sensitivity=[{"delivery_recovery":level,"scenario_sales":round(actual+calculate(level)),"estimated_uplift":round(calculate(level))} for level in levels]
    warnings=[]
    if promotion_uplift>.3: warnings.append("Promotion uplift above 30% is outside the model's common training range.")
    if target_eta<20: warnings.append("Target ETA below 20 minutes may be operationally unrealistic.")
    recommendation="Restore delivery operations first" if delivery_recovery>=.5 else ("Use a targeted promotion" if promotion_uplift>0 else "Increase delivery recovery assumption")
    return {"incident_id":incident_id,"context":{"store_id":event.store_id,"date":str(event.date.date()),"daypart":event.daypart,"channel":event.channel,"current_eta":float(event.delivery_eta),"stockout_detected":bool(event.stockout_flag)},"baseline":{"actual_sales":round(actual),"forecast_sales":round(baseline),"revenue_gap":round(actual-baseline)},"scenario":{"estimated_sales":round(scenario),"estimated_uplift":round(uplift),"remaining_gap":round(remaining),"uplift_low":round(max(0,uplift-uncertainty)),"uplift_high":round(uplift+uncertainty),"confidence":round(confidence,3)},"assumptions":{"delivery_recovery":delivery_recovery,"promotion_uplift":promotion_uplift,"target_eta":target_eta,"stockout_recovery":stockout_recovery},"sensitivity":sensitivity,"recommendation":{"title":recommendation,"reason":"Highest modeled recovery comes from addressing the incident's strongest observed driver before adding discount spend.","confidence":round(confidence,3)},"warnings":warnings,"disclaimer":"Model-estimated scenario, not guaranteed causal impact."}

@app.get("/api/investigate/{store_id}")
def investigation(store_id:str,date:str):
    try: return investigate(store_id.upper(),date)
    except ValueError as exc: raise HTTPException(status_code=404,detail=str(exc)) from exc

@app.get("/api/what-if/{store_id}")
def what_if(store_id:str,date:str,delivery_recovery:float=Query(.5,ge=0,le=1),promotion_uplift:float=Query(0,ge=0,le=.5)):
    try: return simulate(store_id.upper(),date,delivery_recovery,promotion_uplift)
    except ValueError as exc: raise HTTPException(status_code=404,detail=str(exc)) from exc
