import datetime
from typing import Sequence, Tuple

import pandas as pd

from common.time_features import prepare_time_features


class PredictHelper:
    def __init__(self, local_tz_name: str = "Europe/Amsterdam") -> None:
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
        ))

        return df.merge(time_df, on="Timestamp", how="left")
