import json
import os
from datetime import datetime,timezone
import httpx
from src.config import EXTERNAL_SIGNALS_PATH

def _now(): return datetime.now(timezone.utc).isoformat()

def demo_signals():
    return [{"store_id":"S005","signal_type":"delivery_platform","source":"apify_fixture","observed_at":_now(),"platform":"delivery_platform_a","is_available":False,"available_product_ratio":.68,"estimated_delivery_minutes":49,"evidence":"Store listing unavailable during lunch monitoring window"},{"store_id":"S005","signal_type":"review_summary","source":"apify_fixture","observed_at":_now(),"review_count_24h":8,"negative_review_ratio":.75,"average_rating":2.1,"topics":{"delivery_delay":5,"cannot_order":3,"missing_item":2},"sample_reviews":["Delivery was unavailable at lunch","Order arrived late and an item was missing"]},{"store_id":"S005","signal_type":"competitor_context","source":"apify_fixture","observed_at":_now(),"competitor_count_3km":14,"active_promotion_count":3,"average_competitor_rating":4.2,"competitor_free_delivery_count":2}]

def save_signals(items,source="apify"):
    normalized=[]
    for raw in items:
        item=dict(raw); item.setdefault("source",source); item.setdefault("observed_at",_now()); item.setdefault("store_id",item.get("storeId") or "unknown")
        if "signal_type" not in item:
            if any(k in item for k in ("reviewText","review_text","text","stars","rating")): item["signal_type"]="review"
            elif any(k in item for k in ("isAvailable","estimatedDeliveryMinutes","deliveryTime")): item["signal_type"]="delivery_platform"
            else: item["signal_type"]="external_observation"
        normalized.append(item)
    EXTERNAL_SIGNALS_PATH.parent.mkdir(parents=True,exist_ok=True); EXTERNAL_SIGNALS_PATH.write_text(json.dumps(normalized,ensure_ascii=False,indent=2),encoding="utf-8")
    return normalized

def load_signals(store_id=None):
    if not EXTERNAL_SIGNALS_PATH.exists(): save_signals(demo_signals(),"apify_fixture")
    items=json.loads(EXTERNAL_SIGNALS_PATH.read_text(encoding="utf-8"))
    return [x for x in items if not store_id or x.get("store_id")==store_id]

def status():
    items=load_signals(); configured=bool(os.getenv("APIFY_TOKEN") and os.getenv("APIFY_ACTOR_ID"))
    return {"configured":configured,"mode":"live_apify" if configured else "fixture","actor_id":os.getenv("APIFY_ACTOR_ID"),"signal_count":len(items),"last_observed_at":max((x.get("observed_at","") for x in items),default=None)}

def run_configured_actor():
    token=os.getenv("APIFY_TOKEN"); actor=os.getenv("APIFY_ACTOR_ID")
    if not token or not actor: return {"mode":"fixture","items":save_signals(demo_signals(),"apify_fixture")}
    try: run_input=json.loads(os.getenv("APIFY_ACTOR_INPUT_JSON","{}"))
    except json.JSONDecodeError as exc: raise ValueError("APIFY_ACTOR_INPUT_JSON must be valid JSON") from exc
    url=f"https://api.apify.com/v2/actors/{actor.replace('/','~')}/run-sync-get-dataset-items"
    with httpx.Client(timeout=300) as client:
        response=client.post(url,params={"clean":"true","format":"json","timeout":240,"maxItems":500},headers={"Authorization":f"Bearer {token}","Accept":"application/json"},json=run_input); response.raise_for_status(); items=response.json()
    if not isinstance(items,list): raise ValueError("Apify Actor did not return a dataset item array")
    return {"mode":"live_apify","items":save_signals(items,"apify")}

def ingest_webhook(payload):
    items=payload.get("items")
    if not isinstance(items,list): raise ValueError("Webhook payload must include an items array")
    return save_signals(items,"apify_webhook")

def external_context(store_id):
    items=load_signals(store_id)
    return {"store_id":store_id,"freshness":"fixture" if any(x.get("source")=="apify_fixture" for x in items) else "live","signals":items,"summary":{"delivery_unavailable":any(x.get("signal_type")=="delivery_platform" and x.get("is_available") is False for x in items),"negative_review_ratio":next((x.get("negative_review_ratio") for x in items if x.get("signal_type")=="review_summary"),None),"competitor_promotions":next((x.get("active_promotion_count") for x in items if x.get("signal_type")=="competitor_context"),None)}}
