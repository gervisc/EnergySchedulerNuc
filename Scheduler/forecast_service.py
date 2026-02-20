import datetime
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from darts import TimeSeries
from darts.models import LinearRegressionModel, TiDEModel

from common.time_features import FEAT_ORDER, prepare_time_features
from common.db_repository import DbRepository


class ForecastService:
    """Service layer for loading models and producing forecasts."""

    def __init__(
        self,
        repo: DbRepository,
        local_tz_name: str = "Europe/Amsterdam",
    ) -> None:
        self.repo = repo
        self.local_tz_name = local_tz_name

    def load_model(self, model_path: str) -> TiDEModel:
        """Load a saved TiDE model from disk."""
        return TiDEModel.load(model_path)

    def load_solar_model(self, model_path: str) -> LinearRegressionModel:
        """Load a saved solar yield model from disk."""
        path = Path(model_path)
        if path.is_dir():
            path = path / "solar_yield_model"
        return LinearRegressionModel.load(str(path))

    def predict_next_24_hours_consumption(
        self,
        model: Optional[TiDEModel] = None,
        horizon: int = 24,
    ) -> TimeSeries:
        """Predict the next 24 hours (default) from historical data.

        Uses the preparation service to build the last 48 hours of history
        and time-based covariates for history + horizon.
        """
        if model is None:
            model_path = os.environ.get("ENERGYMODELPATH")
            if not model_path:
                raise ValueError("ENERGYMODELPATH is not set; cannot load forecast model")
            model = self.load_model(model_path)

        history = self.prepare_last_48_hours_consumption()
        if not history:
            raise ValueError("No historical consumption data available for prediction")

        df = pd.DataFrame(history, columns=["Timestamp", "WattHour"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        df = df.sort_values("Timestamp").reset_index(drop=True)
        df["WattHour"] = df["WattHour"].astype("float32")

        df["time_idx"] = range(len(df))

        # Build covariate features for history
        history_feats_df = pd.DataFrame(prepare_time_features(
            df["Timestamp"].tolist(),
            local_tz_name=self.local_tz_name,
        ))
        cov_hist_df = df[["Timestamp", "time_idx"]].merge(
            history_feats_df, on="Timestamp", how="left"
        )

        # Build covariate features for future horizon
        last_ts = df["Timestamp"].iloc[-1]
        last_time_idx = int(df["time_idx"].iloc[-1])
        future_ts = [
            last_ts + datetime.timedelta(hours=i)
            for i in range(1, horizon + 1)
        ]
        future_feats_df = pd.DataFrame(prepare_time_features(
            future_ts,
            local_tz_name=self.local_tz_name,
        ))
        future_feats_df["time_idx"] = [
            last_time_idx + i for i in range(1, horizon + 1)
        ]

        cov_full_df = pd.concat(
            [cov_hist_df, future_feats_df],
            ignore_index=True,
            sort=False,
        )
        cov_full_df[FEAT_ORDER] = cov_full_df[FEAT_ORDER].astype("float32")

        series = TimeSeries.from_dataframe(
            df,
            time_col="time_idx",
            value_cols=["WattHour"],
        )

        covariates = TimeSeries.from_dataframe(
            cov_full_df,
            time_col="time_idx",
            value_cols=FEAT_ORDER,
        )

        pred = model.predict(
            n=horizon,
            series=series,
            past_covariates=covariates,
            future_covariates=covariates,
        )

        return pred

    def predict_next_24_hours_solar(
        self,
        model: Optional[LinearRegressionModel] = None,
        horizon: int = 24,
    ) -> TimeSeries:
        """Predict the next 24 hours of solar yield using weather + time features."""
        if model is None:
            model_path = os.environ.get("ENERGYMODELPATH")
            if not model_path:
                raise ValueError("ENERGYMODELPATH is not set; cannot load solar model")
            model = self.load_solar_model(model_path)

        weather_rows = self.repo.get_current_and_next_24_hours_weather()
        if not weather_rows:
            raise ValueError("No weather data available for solar prediction")

        columns = [
            "Timestamp",
            "pressure",
            "temperature",
            "cloud_are_fraction",
            "precipitation",
            "humidity",
            "wind_from_direction",
            "wind_speed",
        ]
        df = pd.DataFrame(weather_rows, columns=columns)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
        df = df.sort_values("Timestamp").reset_index(drop=True)

        time_feats_df = pd.DataFrame(prepare_time_features(
            df["Timestamp"].tolist(),
            local_tz_name=self.local_tz_name,
        ))
        df = df.merge(time_feats_df, on="Timestamp", how="left")

        feature_cols = [
            "pressure",
            "temperature",
            "cloud_are_fraction",
            "precipitation",
            "humidity",
            "wind_from_direction",
            "wind_speed",
            "Sun_Angle",
            "Sun_Angle_Trend",
            "hour_sin",
            "hour_cos",
            "wd_sin",
            "wd_cos",
        ]

        freq = pd.infer_freq(df["Timestamp"]) or "h"
        df = df.set_index("Timestamp").asfreq(freq)
        df = df.interpolate(method="time").ffill().bfill()
        df = df.reset_index()

        feature_frame = df[feature_cols]
        if len(feature_frame) < horizon:
            raise ValueError("Not enough weather rows to produce solar prediction")

        sk_model = getattr(model, "model", None) or getattr(model, "_model", None)
        if sk_model is None:
            raise ValueError("Solar model does not expose an underlying sklearn model")

        y_pred = sk_model.predict(feature_frame.to_numpy())

        pred_df = pd.DataFrame(
            {
                "Timestamp": df["Timestamp"].iloc[: len(y_pred)],
                "solar_total": y_pred,
            }
        )

        return TimeSeries.from_dataframe(
            pred_df,
            time_col="Timestamp",
            value_cols=["solar_total"],
            fill_missing_dates=True,
            freq=freq,
        )
    
    def prepare_last_48_hours_consumption(
        self,
    ) -> List[Tuple[datetime.datetime, float]]:
        """Return a dense 48-hour series of (hour, consumption) in UTC."""
        now = datetime.datetime.now(datetime.timezone.utc)

        end = now.replace(minute=0, second=0, microsecond=0)
        start = end - datetime.timedelta(hours=48)

        raw_rows = self.repo.get_last_48_complete_hours_consumption()
        consumption_by_hour = {
            hour: float(consumption or 0)
            for hour, consumption in raw_rows
        }

        buckets: List[Tuple[datetime.datetime, float]] = []
        current = start
        while current < end:
            buckets.append((current, consumption_by_hour.get(current, 0.0)))
            current += datetime.timedelta(hours=1)

        return buckets
