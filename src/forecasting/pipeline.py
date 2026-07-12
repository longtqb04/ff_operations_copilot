import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.config import FORECAST_HORIZON_DAYS, FORECAST_PATH, MODEL_PATH, SALES_PATH

KEYS=["store_id","daypart","channel","product_category"]
CATS=["store_id","daypart","channel","product_category","region","store_type","promotion_type","delivery_partner"]
NUMS=["weekday","month","week_of_year","promotion_flag","rainfall","temperature","holiday_flag","lag_1","lag_7","lag_14","lag_28","rolling_mean_7","rolling_mean_28"]

def make_features(frame):
    df=frame.copy(); df["date"]=pd.to_datetime(df.date); df=df.sort_values(KEYS+["date"])
    df["weekday"]=df.date.dt.dayofweek; df["month"]=df.date.dt.month; df["week_of_year"]=df.date.dt.isocalendar().week.astype(int)
    group=df.groupby(KEYS,observed=True)["actual_sales"]
    for lag in (1,7,14,28): df[f"lag_{lag}"]=group.shift(lag)
    for window in (7,28): df[f"rolling_mean_{window}"]=group.transform(lambda s:s.shift(1).rolling(window).mean())
    return df.dropna(subset=["lag_28","rolling_mean_28"]).reset_index(drop=True)

def train_and_predict(source=SALES_PATH):
    df=make_features(pd.read_csv(source,low_memory=False)); cutoff=df.date.max()-pd.Timedelta(days=FORECAST_HORIZON_DAYS-1)
    train=df[df.date<cutoff].copy(); test=df[df.date>=cutoff].copy(); columns=CATS+NUMS
    pre=ColumnTransformer([("category",OneHotEncoder(handle_unknown="ignore"),CATS)],remainder="passthrough")
    model=LGBMRegressor(n_estimators=120,learning_rate=.07,num_leaves=31,random_state=42,verbosity=-1,n_jobs=2)
    pipe=Pipeline([("features",pre),("model",model)]); pipe.fit(train[columns],train.actual_sales)
    test["forecast_sales"]=np.maximum(0,pipe.predict(test[columns])).round(); test["residual"]=test.actual_sales-test.forecast_sales
    test["residual_pct"]=test.residual/test.forecast_sales.clip(lower=1); test["absolute_residual"]=test.residual.abs()
    daily=test.groupby(["store_id","date"])[["actual_sales","forecast_sales"]].sum(); daily_error=(daily.actual_sales-daily.forecast_sales).abs()
    metrics={"mape":float((test.absolute_residual/test.actual_sales.clip(lower=1)).mean()),
      "wape":float(test.absolute_residual.sum()/test.actual_sales.sum()),
      "rmse":float(mean_squared_error(test.actual_sales,test.forecast_sales)**.5),
      "bias":float(test.residual.sum()/test.actual_sales.sum()),
      "daily_store_mape":float((daily_error/daily.actual_sales.clip(lower=1)).mean()),
      "daily_store_wape":float(daily_error.sum()/daily.actual_sales.sum())}
    FORECAST_PATH.parent.mkdir(parents=True,exist_ok=True); MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    test.to_csv(FORECAST_PATH,index=False); joblib.dump({"pipeline":pipe,"columns":columns,"metrics":metrics},MODEL_PATH); return metrics
