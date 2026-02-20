import datetime
import math
from typing import Any, Dict, Iterable, List, Optional

import pytz

FEAT_ORDER = [
    "Sun_Angle",
    "Sun_Angle_Trend",
    "wd_sin",
    "wd_cos",
    "hour_cos",
    "hour_sin",
]


def calculate_sun_angle(
    timestamp: datetime.datetime,
    longitude: float = 4.70,
    latitude: float = 52.01,
) -> float:
    """Return sun angle (radians), clipped at 0 below horizon."""
    day_of_year = timestamp.timetuple().tm_yday

    declination = 23.45 * 2 * math.pi / 360 * math.sin(
        2 * math.pi * (day_of_year - 81) / 365
    )
    if day_of_year < 107:
        eqt = -14.2 * math.sin(math.pi * (day_of_year + 7) / 111)
    elif day_of_year < 167:
        eqt = 4 * math.sin(math.pi * (day_of_year - 106) / 59)
    elif day_of_year < 247:
        eqt = 6.5 * math.sin(math.pi * (day_of_year - 166) / 80)
    else:
        eqt = 16.4 * math.sin(math.pi * (day_of_year - 247) / 113)

    t_solar = timestamp.hour + eqt / 60 - longitude / 15

    hour_angle = math.pi * (12 - t_solar) / 12
    altitude = math.asin(
        math.sin(math.radians(latitude)) * math.sin(declination)
        + math.cos(math.radians(latitude))
        * math.cos(declination)
        * math.cos(hour_angle)
    )
    return max(altitude, 0)


def prepare_time_features(
    utc_datetimes: Iterable[datetime.datetime],
    local_tz_name: str = "Europe/Amsterdam",
    longitude: float = 4.70,
    latitude: float = 52.01,
) -> List[Dict[str, Any]]:
    """Return per-timestamp features from UTC datetimes.

    Each entry contains:
    Timestamp, Sun_Angle, Sun_Angle_Trend, hour_sin, hour_cos, wd_sin, wd_cos
    """
    local_tz = pytz.timezone(local_tz_name)
    results: List[Dict[str, Any]] = []

    previous_angle: Optional[float] = None
    for ts in utc_datetimes:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        else:
            ts = ts.astimezone(datetime.timezone.utc)

        local_ts = ts.astimezone(local_tz)
        sun_angle = calculate_sun_angle(
            ts,
            longitude=longitude,
            latitude=latitude,
        )

        if previous_angle is None:
            sun_angle_trend = -1
        else:
            sun_angle_trend = 1 if (sun_angle - previous_angle) > 0 else -1

        hour_angle = 2 * math.pi * local_ts.hour / 24
        hour_sin = math.sin(hour_angle)
        hour_cos = math.cos(hour_angle)

        wd_angle = 2 * math.pi * local_ts.weekday() / 7
        wd_sin = math.sin(wd_angle)
        wd_cos = math.cos(wd_angle)

        results.append(
            {
                "Timestamp": ts,
                "Sun_Angle": sun_angle,
                "Sun_Angle_Trend": sun_angle_trend,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "wd_sin": wd_sin,
                "wd_cos": wd_cos,
            }
        )
        previous_angle = sun_angle

    return results
