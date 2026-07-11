import json
import pandas as pd
from src.config import ANOMALY_PATH, FORECAST_PATH
from src.external.apify import external_context

def _event(incident_id):
    a=pd.read_csv(ANOMALY_PATH); a["date"]=pd.to_datetime(a.date)
    a["incident_id"]=a.apply(lambda r:f"{r.store_id}-{r.date:%Y%m%d}-{r.daypart}-{r.channel}",axis=1)
    m=a[a.incident_id==incident_id]
    if m.empty: raise ValueError("Incident not found")
    return m.iloc[0]

def get_anomaly_event(incident_id,**_):
    e=_event(incident_id)
    return {"store_id":e.store_id,"date":str(e.date.date()),"region":e.region,"daypart":e.daypart,"channel":e.channel,"severity":e.severity,"actual_sales":int(e.actual_sales),"forecast_sales":int(e.forecast_sales),"revenue_gap":int(e.residual),"residual_pct":round(float(e.residual_pct),4),"anomaly_score":round(float(e.anomaly_score),4)}

def _day(incident_id):
    e=_event(incident_id); df=pd.read_csv(FORECAST_PATH); df["date"]=pd.to_datetime(df.date)
    return e,df[(df.store_id==e.store_id)&(df.date==e.date)].copy(),df

def get_channel_breakdown(incident_id,**_):
    _,day,_=_day(incident_id); x=day.groupby("channel")[["actual_sales","forecast_sales","transaction_count"]].sum(); x["revenue_gap"]=x.actual_sales-x.forecast_sales; x["variance_pct"]=x.revenue_gap/x.forecast_sales.clip(lower=1)
    return json.loads(x.reset_index().to_json(orient="records"))

def get_daypart_breakdown(incident_id,**_):
    _,day,_=_day(incident_id); x=day.groupby("daypart")[["actual_sales","forecast_sales","transaction_count"]].sum(); x["revenue_gap"]=x.actual_sales-x.forecast_sales; x["variance_pct"]=x.revenue_gap/x.forecast_sales.clip(lower=1)
    return json.loads(x.reset_index().to_json(orient="records"))

def get_peer_comparison(incident_id,**_):
    e,_,df=_day(incident_id); peers=df[(df.date==e.date)&(df.region==e.region)&(df.store_id!=e.store_id)]
    x=peers.groupby("store_id")[["actual_sales","forecast_sales"]].sum(); variance=(x.actual_sales-x.forecast_sales)/x.forecast_sales.clip(lower=1)
    return {"store_variance_pct":round(float(e.residual_pct),4),"peer_median_variance_pct":round(float(variance.median()),4),"peer_count":len(x),"assessment":"store_specific" if e.residual_pct<variance.median()-.08 else "regional_or_market"}

def get_operational_signals(incident_id,**_):
    e,day,_=_day(incident_id); focus=day[(day.daypart==e.daypart)&(day.channel==e.channel)].iloc[0]
    return {"delivery_eta":{"observed":float(focus.delivery_eta),"baseline":27,"delta":round(float(focus.delivery_eta-27),1)},"stockout":{"observed":bool(focus.stockout_flag)},"transactions":{"observed":int(focus.transaction_count),"estimated_last_week":round(float(focus.lag_7/max(focus.average_order_value,1))),"change_pct":round(float(focus.transaction_count/max(focus.lag_7/max(focus.average_order_value,1),1)-1),3)},"weather":{"rainfall":float(focus.rainfall),"temperature":float(focus.temperature)},"promotion_active":bool(focus.promotion_flag)}

def get_recommended_actions(incident_id,**_):
    e=_event(incident_id); actions=[{"action":"Verify delivery platform availability and integration","owner":"Store manager","confidence":.88,"impact_low":int(e.absolute_residual*.28),"impact_high":int(e.absolute_residual*.48),"effort":"low"},{"action":"Restore delivery ETA through staffing and kitchen throughput checks","owner":"Operations lead","confidence":.82,"impact_low":int(e.absolute_residual*.20),"impact_high":int(e.absolute_residual*.38),"effort":"medium"}]
    if e.stockout_flag: actions.append({"action":"Restore top lunch product availability","owner":"Inventory lead","confidence":.76,"impact_low":int(e.absolute_residual*.10),"impact_high":int(e.absolute_residual*.22),"effort":"low"})
    return actions

def run_what_if(incident_id,delivery_recovery=.5,promotion_uplift=0,**_):
    e=_event(incident_id); gap=max(0,float(e.forecast_sales-e.actual_sales)); uplift=gap*max(0,min(float(delivery_recovery),1))*.72+float(e.forecast_sales)*max(0,min(float(promotion_uplift),.5))
    return {"assumptions":{"delivery_recovery":delivery_recovery,"promotion_uplift":promotion_uplift},"estimated_uplift":round(uplift),"estimated_sales":round(float(e.actual_sales)+uplift),"remaining_gap":round(float(e.forecast_sales-e.actual_sales-uplift)),"disclaimer":"Model estimate, not guaranteed causal impact."}

def get_external_signals(incident_id,**_):
    return external_context(_event(incident_id).store_id)

TOOL_HANDLERS={"get_anomaly_event":get_anomaly_event,"get_channel_breakdown":get_channel_breakdown,"get_daypart_breakdown":get_daypart_breakdown,"get_peer_comparison":get_peer_comparison,"get_operational_signals":get_operational_signals,"get_external_signals":get_external_signals,"get_recommended_actions":get_recommended_actions,"run_what_if":run_what_if}
TOOL_DEFINITIONS=[{"type":"function","name":name,"description":desc,"parameters":params,"strict":True} for name,desc,params in [
 ("get_anomaly_event","Get the anomaly's core metrics and business impact.",{"type":"object","properties":{},"additionalProperties":False}),
 ("get_channel_breakdown","Compare actual and forecast performance by sales channel.",{"type":"object","properties":{},"additionalProperties":False}),
 ("get_daypart_breakdown","Compare actual and forecast performance by daypart.",{"type":"object","properties":{},"additionalProperties":False}),
 ("get_peer_comparison","Compare the store with regional peer stores.",{"type":"object","properties":{},"additionalProperties":False}),
 ("get_operational_signals","Get ETA, stock-out, transaction, weather, and promotion evidence.",{"type":"object","properties":{},"additionalProperties":False}),
 ("get_external_signals","Get normalized Apify delivery-platform, review sentiment, and competitor intelligence.",{"type":"object","properties":{},"additionalProperties":False}),
 ("get_recommended_actions","Get evidence-aligned actions from the governed action library.",{"type":"object","properties":{},"additionalProperties":False}),
 ("run_what_if","Estimate recovery for a bounded operational scenario.",{"type":"object","properties":{"delivery_recovery":{"type":"number","minimum":0,"maximum":1},"promotion_uplift":{"type":"number","minimum":0,"maximum":.5}},"required":["delivery_recovery","promotion_uplift"],"additionalProperties":False})]]

def execute_tool(name,incident_id,arguments):
    if name not in TOOL_HANDLERS: raise ValueError(f"Unknown tool: {name}")
    return TOOL_HANDLERS[name](incident_id,**arguments)
