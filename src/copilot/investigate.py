import pandas as pd
from src.config import FORECAST_PATH

def investigate(store_id, date, source=FORECAST_PATH):
    df = pd.read_csv(source)
    df["date"] = pd.to_datetime(df.date)
    target = pd.Timestamp(date)
    store=df[(df.store_id==store_id)&(df.date==target)].copy()
    if store.empty:
       raise ValueError("No forecast data for this store and date")
    
    actual = store.actual_sales.sum()
    forecast = store.forecast_sales.sum()
    gap = actual - forecast
    channel = store.groupby("channel")[["actual_sales","forecast_sales"]].sum(); channel["gap"]=channel.actual_sales-channel.forecast_sales
    daypart = store.groupby("daypart")[["actual_sales","forecast_sales"]].sum(); daypart["gap"]=daypart.actual_sales-daypart.forecast_sales
    peers=df[(df.date==target)&(df.region==store.region.iloc[0])&(df.store_id!=store_id)].groupby("store_id")[["actual_sales","forecast_sales"]].sum()
    peer_delta=((peers.actual_sales-peers.forecast_sales)/peers.forecast_sales.clip(lower=1)).median() if len(peers) else 0
    worst_channel=channel.gap.idxmin(); worst_daypart=daypart.gap.idxmin()
    subset=store[(store.channel==worst_channel)&(store.daypart==worst_daypart)]
    eta=float(subset.delivery_eta.mean()); residual_pct=gap/max(forecast,1); actions=[]
    if worst_channel=="delivery": actions.append({"action":"Verify delivery platform availability and integration","priority":1,"confidence":.88})
    if eta>35: actions.append({"action":"Restore delivery ETA by checking staffing and kitchen throughput","priority":2,"confidence":.82})
    if store.stockout_flag.max()>0: actions.append({"action":"Review availability of top lunch products","priority":3,"confidence":.76})
    if not actions: actions.append({"action":"Contact store manager and verify operational disruptions","priority":1,"confidence":.65})
    return {"store_id":store_id,"date":str(target.date()),"actual_sales":int(actual),"forecast_sales":int(forecast),"revenue_gap":int(gap),
      "residual_pct":round(float(residual_pct),4),"peer_median_residual_pct":round(float(peer_delta),4),
      "scope_assessment":"store_specific" if residual_pct<peer_delta-.08 else "regional_or_market",
      "drivers":[{"driver":f"{worst_channel} channel","revenue_gap":int(channel.loc[worst_channel,"gap"])},{"driver":f"{worst_daypart} daypart","revenue_gap":int(daypart.loc[worst_daypart,"gap"])},{"driver":"delivery ETA","observed":round(eta,1),"unit":"minutes"}],
      "recommended_actions":actions,"executive_summary":f"{store_id} recorded sales {abs(residual_pct):.1%} {'below' if gap<0 else 'above'} forecast. The largest gap was in {worst_channel} during {worst_daypart}; peer stores were {peer_delta:+.1%} versus forecast.",
      "disclaimer":"Drivers are evidence-based associations; they are not guaranteed causal effects."}

def simulate(store_id,date,delivery_recovery=0.,promotion_uplift=0.):
    report=investigate(store_id,date); baseline=report["forecast_sales"]; recoverable=max(0,-report["revenue_gap"])
    uplift=recoverable*max(0,min(delivery_recovery,1))+baseline*max(0,min(promotion_uplift,.5))
    return {"baseline_forecast":baseline,"scenario_sales":round(baseline+uplift),"estimated_uplift":round(uplift),
      "inputs":{"delivery_recovery":delivery_recovery,"promotion_uplift":promotion_uplift},"disclaimer":"Model-estimated scenario, not guaranteed causal impact."}
