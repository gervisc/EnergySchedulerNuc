import datetime
import logging
import os
import re
import unicodedata
from typing import Optional, Tuple, List, TYPE_CHECKING

from sqlalchemy import and_, create_engine, func, text
from sqlalchemy.orm import Session, sessionmaker

from common.entities import Base, WeatherRecord, AnkerData, AnkerDataOld,  EnergyPrice, Scheduler
from common.constants import CONSUMPTION_FACTOR

if TYPE_CHECKING:
    from common.anker_repository import AnkerDailyCounters

DEFAULT_DB_ENV_VAR = "energydb"


def get_db_connection_string(env_var: str) -> str:
    value = os.environ.get(env_var)
    if value:
        return value
    raise ValueError(f"Database connection string is not configured (env {env_var} missing)")


class DbRepository:
    """Repository for database operations backed by SQLAlchemy."""

    def __init__(self, connection_string: str, logger: logging.Logger) -> None:
        self.logger = logger
        self.connection_string = connection_string
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

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

        Consumption is read from AnkerData.home_consumption.
        """
        now = datetime.datetime.now(datetime.timezone.utc)

        end = now.replace(minute=0, second=0, microsecond=0)
        start = end - datetime.timedelta(hours=48)

        hour_bucket = func.from_unixtime(
            func.floor(func.unix_timestamp(AnkerData.Timestamp) / 3600) * 3600
        )
        query = (
            self.session.query(
                hour_bucket.label("hour"),
                func.sum(func.coalesce(AnkerData.home_consumption, 0)).label("consumption"),
            )
            .filter(
                AnkerData.Timestamp >= start,
                AnkerData.Timestamp < end,
            )
            .group_by(hour_bucket)
            .order_by(hour_bucket)
        )

        try:
            self.logger.info(
                "Running AnkerData consumption query (start=%s, end=%s)",
                start,
                end,
            )
            rows = query.all()
        except Exception:
            self.logger.exception(
                "Failed to load last 48 hours AnkerData consumption (start=%s, end=%s)",
                start,
                end,
            )
            raise
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

    def get_current_and_next_24_hours_prices(self):
        """Return the current electricity price and all future prices (UTC)."""
        now = datetime.datetime.now(datetime.timezone.utc)
        start = now.replace(minute=0, second=0, microsecond=0)

        future_rows = (
            self.session.query(
                EnergyPrice.timestamp,
                EnergyPrice.price,
                EnergyPrice.price_excl_tax,
            )
            .filter(
                EnergyPrice.timestamp >= start,
            )
            .order_by(EnergyPrice.timestamp)
            .all()
        )

        future = [
            (
                timestamp.replace(tzinfo=datetime.timezone.utc),
                price,
                price_excl_tax,
            )
            for timestamp, price, price_excl_tax in future_rows
        ]

        return future

    def get_current_battery_state(self):
        """Return the latest battery state of charge from AnkerData (UTC)."""
        row = (
            self.session.query(AnkerData.Timestamp, AnkerData.battery_health)
            .filter(AnkerData.battery_health.isnot(None))
            .order_by(AnkerData.Timestamp.desc())
            .first()
        )
        if not row:
            return None
        ts, value = row
        return ts.replace(tzinfo=datetime.timezone.utc), value

    def upsert_scheduler_rows(
        self,
        rows: List[
            Tuple[
                datetime.datetime,
                Optional[bool],
                Optional[bool],
                Optional[bool],
                Optional[float],
                Optional[float],
                Optional[float],
                Optional[float],
            ]
        ],
    ) -> None:
        """Upsert scheduler rows and delete entries older than 48 hours (UTC)."""
        if rows:
            min_ts = min(row[0] for row in rows)
            self.session.query(Scheduler).filter(Scheduler.timestamp >= min_ts).delete(
                synchronize_session=False
            )
            objects = [
                Scheduler(
                    timestamp=timestamp,
                    charge_on=charge_on,
                    solar_charge_on=solar_charge_on,
                    discharge_on=discharge_on,
                    solar=solar,
                    consumption=consumption,
                    grid=grid,
                    battery=battery,
                )
                for (
                    timestamp,
                    charge_on,
                    solar_charge_on,
                    discharge_on,
                    solar,
                    consumption,
                    grid,
                    battery,
                ) in rows
            ]
            self.session.add_all(objects)

        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=48)
        self.session.query(Scheduler).filter(Scheduler.timestamp < cutoff).delete(
            synchronize_session=False
        )
        self.session.commit()

    def get_expected_discharge_sell_price(self) -> Optional[float]:
        """Return weighted-average energy price for discharged energy over last 7 days (UTC)."""
        now = datetime.datetime.now(datetime.timezone.utc)
        end = now.replace(minute=0, second=0, microsecond=0)
        start = end - datetime.timedelta(days=7)

        discharge_hour = func.from_unixtime(
            func.floor(func.unix_timestamp(AnkerData.Timestamp) / 3600) * 3600
        ).label("hour")

        discharge_subq = (
            self.session.query(
                discharge_hour,
                func.sum(func.coalesce(AnkerData.discharge, 0)).label("discharge_kwh"),
            )
            .filter(AnkerData.Timestamp >= start)
            .group_by(discharge_hour)
            .subquery()
        )

        row = (
            self.session.query(
                func.sum(discharge_subq.c.discharge_kwh * EnergyPrice.price).label("weighted_sum"),
                func.sum(discharge_subq.c.discharge_kwh).label("total_discharge"),
            )
            .join(EnergyPrice, EnergyPrice.timestamp == discharge_subq.c.hour)
            .first()
        )

        if not row:
            return None
        weighted_sum, total_discharge = row
        if not total_discharge:
            avg_price = (
                self.session.query(func.avg(EnergyPrice.price))
                .filter(EnergyPrice.timestamp >= start)
                .scalar()
            )
            return float(avg_price) if avg_price is not None else None
        return float(weighted_sum / total_discharge)

    def upsert_anker_daily_counters(
        self,
        timestamp: datetime.datetime,
        counters: "AnkerDailyCounters",
    ) -> None:
        """Upsert one Anker row for a given timestamp in AnkerData."""

        row = self.session.get(AnkerData, timestamp)
        if row is None:
            row = AnkerData(Timestamp=timestamp)
            self.session.add(row)

        row.battery_diff = counters.state_of_charge
        if counters.current_soc is not None and counters.state_of_charge is not None:
            row.battery_health = float(counters.current_soc) - float(counters.state_of_charge)
        else:
            row.battery_health = None
        row.grid_export = counters.grid_export
        row.grid_import = counters.grid_import
        row.home_consumption = counters.home_consumption
        row.discharge = counters.discharge
        row.grid_charge = counters.grid_charge
        row.solar_total = counters.solar_yield
        row.solar_charge = counters.solar_charge

        self.session.commit()
        
