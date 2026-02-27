import os
import sys

import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

try:
    from common.time_features import (
        DEFAULT_LATITUDE,
        DEFAULT_LOCAL_TZ_NAME,
        DEFAULT_LONGITUDE,
        FEAT_ORDER,
        prepare_time_features,
    )
except ModuleNotFoundError:
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from common.time_features import (
        DEFAULT_LATITUDE,
        DEFAULT_LOCAL_TZ_NAME,
        DEFAULT_LONGITUDE,
        FEAT_ORDER,
        prepare_time_features,
    )

def pepare_data(df):
    df = df.copy()
    # Normalize so Timestamp is a single column (not also index)
   
    df["time_idx"] = pd.factorize(df["Timestamp"])[0].astype(int)

    df = df.assign(id="electricity")

    time_features = prepare_time_features(
        df["Timestamp"].tolist(),
        local_tz_name=DEFAULT_LOCAL_TZ_NAME,
        longitude=DEFAULT_LONGITUDE,
        latitude=DEFAULT_LATITUDE,
    )

  
    time_df = pd.DataFrame(time_features)

    df = df.merge(time_df, on="Timestamp", how="left")

    cols_fp32 = [
    "WattHour", "Sun_Angle", "Sun_Angle_Trend",
    "wd_sin", "wd_cos", "hour_cos", "hour_sin",
    ]
    df[cols_fp32] = df[cols_fp32].astype("float32")
    return df

def preparation_data_darts(df):
    series= TimeSeries.from_dataframe(
    df,
    time_col="time_idx",
    value_cols=["WattHour"],  # ← covariates
    )

    series_cov= TimeSeries.from_dataframe(
        df,
        time_col="time_idx",
        value_cols=[  *FEAT_ORDER],  # ← covariates
    )

    # one scaler for the target
    sc_tgt = Scaler()
    series_scaled = series# sc_tgt.fit_transform(series)          # 1 component

    # another scaler for the covariates
    sc_cov = Scaler()
    cov_scaled = series_cov#sc_cov.fit_transform(series_cov) 
    return series_scaled, cov_scaled, sc_tgt, sc_cov
