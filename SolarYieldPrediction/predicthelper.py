import datetime
from typing import Sequence, Tuple

import pandas as pd

from common.time_features import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, prepare_time_features


class PredictHelper:
    def __init__(self, local_tz_name: str) -> None:
        self.local_tz_name = local_tz_name

    def build_feature_dataframe(
        self,
        hourly_rows: Sequence[
            Tuple[
                datetime.datetime,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
            ]
        ],
    ) -> pd.DataFrame:
        """Build a dataframe with weather, solar, timestamp, and time features."""
        columns = [
            "Timestamp",
            "solar_total",
            "pressure",
            "temperature",
            "cloud_are_fraction",
            "precipitation",
            "humidity",
            "wind_from_direction",
            "wind_speed",
        ]
        df = pd.DataFrame(hourly_rows, columns=columns)

        time_df = pd.DataFrame(prepare_time_features(
            df["Timestamp"].tolist(),
            local_tz_name=self.local_tz_name,
            longitude=DEFAULT_LONGITUDE,
            latitude=DEFAULT_LATITUDE,
        ))

        return df.merge(time_df, on="Timestamp", how="left")
