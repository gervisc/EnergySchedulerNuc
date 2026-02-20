import datetime
import logging
import os
import re
import unicodedata
from typing import Optional, Tuple, List

from sqlalchemy import and_, create_engine, func, text
from sqlalchemy.orm import Session, sessionmaker

from common.entities import Base, WeatherRecord, AnkerData, AnkerDataOld, DailyCounter


class DbRepository:
    """Repository for database operations backed by SQLAlchemy."""

    def __init__(self, connection_string: Optional[str] = None, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.connection_string = connection_string or os.environ.get("energydb")
        if not self.connection_string:
            raise ValueError("Database connection string is not configured (env ENERGYDB missing)")

        self.engine = create_engine(self.connection_string)
        Base.metadata.create_all(self.engine)

        self._session_factory = sessionmaker(bind=self.engine)
        self.session: Session = self._session_factory()

    def close(self) -> None:
        """Close the underlying session. Call when you're done with the repository."""
        self.session.close()

    def __enter__(self) -> "DbRepository":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get_last_48_complete_hours_consumption(self):
        """Return hourly power consumption for the last 48 complete hours.

        Consumption = solar_to_home_total + grid_to_home_total + battery_to_home_total.
        """
       
        now = datetime.datetime.now(datetime.timezone.utc)

        end = now.replace(minute=0, second=0, microsecond=0)
        start = end - datetime.timedelta(hours=48)

        entity_ids = [
            "sensor.system_pad3400_daily_home_usage",
     ]

        query = (
            self.session.query(
                DailyCounter.bucket.label("hour"),
                func.sum(func.coalesce(DailyCounter.delta, 0)*1000).label("consumption"),
            )
            .filter(
                DailyCounter.bucket >= start,
                DailyCounter.bucket < end,
                DailyCounter.entity_id.in_(entity_ids),
            )
            .group_by(DailyCounter.bucket)
            .order_by(DailyCounter.bucket)
        )

        rows = query.all()
        return [(hour.replace(tzinfo=datetime.timezone.utc), consumption) for hour, consumption in rows]

        

    def get_hourly_solar_with_weather(self):
        """Return hourly solar totals joined with hourly-aggregated weather data.

        Solar total = solar_to_battery_total + solar_to_grid_total + solar_to_home_total.
        """
        solar_hour = func.from_unixtime(
            func.floor(func.unix_timestamp(AnkerDataOld.Timestamp) / 3600) * 3600
        ).label("hour")
        weather_hour = func.from_unixtime(
            func.floor(func.unix_timestamp(WeatherRecord.timestamp) / 3600) * 3600
        ).label("hour")

        solar_total = func.sum(
            func.coalesce(AnkerDataOld.solar_to_battery_total, 0)
            + func.coalesce(AnkerDataOld.solar_to_grid_total, 0)
            + func.coalesce(AnkerDataOld.solar_to_home_total, 0)
        ).label("solar_total")

        solar_subq = (
            self.session.query(
                solar_hour,
                solar_total,
            )
            .group_by(solar_hour)
            .subquery()
        )

        weather_subq = (
            self.session.query(
                weather_hour,
                func.avg(WeatherRecord.pressure).label("pressure"),
                func.avg(WeatherRecord.temperature).label("temperature"),
                func.avg(WeatherRecord.cloud_are_fraction).label("cloud_are_fraction"),
                func.avg(WeatherRecord.precipitation).label("precipitation"),
                func.avg(WeatherRecord.humidity).label("humidity"),
                func.avg(WeatherRecord.wind_from_direction).label("wind_from_direction"),
                func.avg(WeatherRecord.wind_speed).label("wind_speed"),
            )
            .group_by(weather_hour)
            .subquery()
        )

        query = (
            self.session.query(
                solar_subq.c.hour.label("hour"),
                solar_subq.c.solar_total,
                weather_subq.c.pressure,
                weather_subq.c.temperature,
                weather_subq.c.cloud_are_fraction,
                weather_subq.c.precipitation,
                weather_subq.c.humidity,
                weather_subq.c.wind_from_direction,
                weather_subq.c.wind_speed,
            )
            .join(weather_subq, solar_subq.c.hour == weather_subq.c.hour)
            .order_by(solar_subq.c.hour)
        )

        rows = query.all()
        return [
            (
                hour.replace(tzinfo=datetime.timezone.utc),
                solar_total,
                pressure,
                temperature,
                cloud_are_fraction,
                precipitation,
                humidity,
                wind_from_direction,
                wind_speed,
            )
            for (
                hour,
                solar_total,
                pressure,
                temperature,
                cloud_are_fraction,
                precipitation,
                humidity,
                wind_from_direction,
                wind_speed,
            ) in rows
        ]

    def get_current_and_next_24_hours_weather(self):
        """Return hourly-aggregated weather data for the current hour plus next 23 hours (UTC).

        Hours are bucketed to the start of the hour in UTC.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        start = now.replace(minute=0, second=0, microsecond=0)
        end = start + datetime.timedelta(hours=24)

        weather_hour = func.from_unixtime(
            func.floor(func.unix_timestamp(WeatherRecord.timestamp) / 3600) * 3600
        ).label("hour")

        query = (
            self.session.query(
                weather_hour,
                func.avg(WeatherRecord.pressure).label("pressure"),
                func.avg(WeatherRecord.temperature).label("temperature"),
                func.avg(WeatherRecord.cloud_are_fraction).label("cloud_are_fraction"),
                func.avg(WeatherRecord.precipitation).label("precipitation"),
                func.avg(WeatherRecord.humidity).label("humidity"),
                func.avg(WeatherRecord.wind_from_direction).label("wind_from_direction"),
                func.avg(WeatherRecord.wind_speed).label("wind_speed"),
            )
            .filter(
                WeatherRecord.timestamp >= start,
                WeatherRecord.timestamp < end,
            )
            .group_by(weather_hour)
            .order_by(weather_hour)
        )

        rows = query.all()
        return [
            (
                hour.replace(tzinfo=datetime.timezone.utc),
                pressure,
                temperature,
                cloud_are_fraction,
                precipitation,
                humidity,
                wind_from_direction,
                wind_speed,
            )
            for (
                hour,
                pressure,
                temperature,
                cloud_are_fraction,
                precipitation,
                humidity,
                wind_from_direction,
                wind_speed,
            ) in rows
        ]
        
