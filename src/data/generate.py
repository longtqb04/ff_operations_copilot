"""Deterministic synthetic POS data for a self-contained demo."""
import argparse
import numpy as np
import pandas as pd
from src.config import RANDOM_SEED, SALES_PATH

def generate_sales(days=240, stores=12, output=SALES_PATH, seed=RANDOM_SEED, shuffle=False):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2026-07-10")
    dates = pd.date_range(end=end, periods=days)
    rows = []
    for n in range(1, stores + 1):
        sid = f"S{n:03d}"; region = ["HCM", "Hanoi", "Danang"][n % 3]
        store_type = "mall" if n % 2 == 0 else "street"; base = rng.uniform(7e6, 11e6)
        for date in dates:
            weekend = 1.16 if date.dayofweek >= 5 else 1.0
            trend = 1 + (date - dates[0]).days * .0007
            rain = max(0, rng.normal(5 if 5 <= date.month <= 10 else 2, 5))
            temp = rng.normal(29, 2.5); promo = int(rng.random() < .18)
            holiday = int((date.month, date.day) in {(1, 1), (4, 30), (5, 1)})
            for daypart, dpf in {"breakfast":.55, "lunch":1.15, "dinner":1.3}.items():
                event_time=date+pd.Timedelta(hours={"breakfast":8,"lunch":12,"dinner":19}[daypart])
                for channel, chf in {"dine_in":1.,"takeaway":.7,"kiosk":.58,"delivery":.88,"app":.64}.items():
                    incident = sid == "S005" and date >= end-pd.Timedelta(days=4) and daypart == "lunch" and channel == "delivery"
                    weather = 1 + (.012*rain if channel == "delivery" else -.005*rain)
                    channel_sales=base*dpf*chf*weekend*trend*weather*(1.2 if promo else 1)*rng.normal(1,.055)
                    if incident: channel_sales*=.48
                    product_mix={"chicken":.34,"burger":.16,"rice":.14,"sides":.10,"beverage":.10,"dessert":.06,"combo":.10}
                    for category,share in product_mix.items():
                        sales=max(1,channel_sales*share*rng.normal(1,.04)); stockout=int(incident and category in {"chicken","combo"})
                        if stockout: sales*=.72
                        item_price={"chicken":82000,"burger":65000,"rice":59000,"sides":35000,"beverage":28000,"dessert":32000,"combo":145000}[category]
                        quantity=max(1,round(sales/item_price)); tx=max(1,round(quantity*rng.uniform(.72,.95)))
                        rows.append({"store_id":sid,"date":date,"event_timestamp":event_time,"daypart":daypart,"channel":channel,"product_category":category,
                          "region":region,"store_type":store_type,"actual_sales":round(sales),"quantity":quantity,
                          "transaction_count":tx,"average_order_value":round(sales/tx),"promotion_flag":promo,
                          "promotion_type":"bundle" if promo else "none","rainfall":round(rain,2),"temperature":round(temp,2),
                          "holiday_flag":holiday,"delivery_partner":"aggregator_a" if channel=="delivery" else "owned_or_na",
                          "delivery_eta":round(rng.normal(47 if incident else 27,2.5),1),"stockout_flag":stockout,
                          "is_actionable_anomaly":incident,"incident_started_at":event_time if incident else pd.NaT,
                          "data_available_at":event_time+pd.Timedelta(minutes=35 if incident else 45)})
    df=pd.DataFrame(rows).sort_values(["store_id","daypart","channel","product_category","date"])
    if shuffle:
        df=df.sample(frac=1,random_state=seed).reset_index(drop=True)
    output.parent.mkdir(parents=True,exist_ok=True); df.to_csv(output,index=False); return df

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--days",type=int,default=240); p.add_argument("--stores",type=int,default=12)
    a=p.parse_args(); result=generate_sales(a.days,a.stores); print(f"Generated {len(result):,} rows at {SALES_PATH}")
