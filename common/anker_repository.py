from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

from requests import Session

from common.anker_api.api import AnkerSolixApi
from common.anker_api.apitypes import SolarbankUsageMode, SolixTariffTypes

STATE_FILENAME = "anker_metrics_state.json"


class AnkerRepository:
    """Repository for Anker API queries."""

    def __init__(
        self,
        user: str,
        password: str,
        country: str,
        site_id: str,
        device_sn: str,
        logger: logging.Logger,
    ) -> None:
        self.logger = logger
        self.user = user
        self.password = password
        self.country = country
        self.site_id = site_id
        self.device_sn = device_sn
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def get_current_metrics(self, timestep_minutes: int) -> "AnkerMetrics":
        """Return current SOC and diffs for today's energy counters."""
        previous_state = self._load_state()
        if previous_state:
            timestamp = previous_state.get("timestamp")
            if timestamp:
                try:
                    saved_ts = datetime.fromisoformat(str(timestamp))
                except ValueError:
                    saved_ts = None
                if saved_ts is not None:
                    now_local = datetime.now().astimezone()
                    if saved_ts.tzinfo is None:
                        saved_ts = saved_ts.replace(tzinfo=now_local.tzinfo)
                    age = now_local - saved_ts.astimezone(now_local.tzinfo)
                    if age < timedelta(minutes=timestep_minutes * 0.5):
                        raise ValueError(
                            "No previous Anker metrics state found; cannot calculate diffs yet."
                        )

        with Session() as websession:
            api = AnkerSolixApi(
                self.user,
                self.password,
                self.country,
                websession,
                self.logger,
            )
            api.authenticate()
            today = datetime.today()

            solar_data = api.energy_analysis(
                siteId=self.site_id,
                deviceSn=self.device_sn,
                startDay=today,
                endDay=today,
                rangeType="day",
                devType="solar_production",
            )

            home_data = api.energy_analysis(
                siteId=self.site_id,
                deviceSn=self.device_sn,
                startDay=today,
                endDay=today,
                rangeType="day",
                devType="home_usage",
            )
            grid_data = api.energy_analysis(
                siteId=self.site_id,
                deviceSn=self.device_sn,
                startDay=today,
                endDay=today,
                rangeType="day",
                devType="grid",
            )

            def _to_float(value: object, default: float = 0.0) -> float:
                try:
                    return float(value) if value is not None else default
                except (TypeError, ValueError):
                    return default

            scene_data = api.get_scene_info(siteId=self.site_id)
            current_soc = _to_float(
                scene_data["solarbank_info"]["solarbank_list"][0]["battery_power"]
            )
            solar_charge = _to_float(solar_data["solar_to_battery_total"])
            solar_yield = _to_float(solar_data["solar_total"])
            solar_grid_charge = _to_float(solar_data["solar_to_grid_total"])
            grid_charge = _to_float(grid_data["grid_to_battery_total"])
            discharge = _to_float(home_data["battery_discharging_total"])
            grid_export = _to_float(grid_data["solar_to_grid_total"])
            grid_home = _to_float(home_data["grid_to_home_total"])
            home_consumption = _to_float(home_data["home_usage_total"])
            state_of_charge = current_soc

            current_values = {
                "solar_charge": solar_charge,
                "solar_yield": solar_yield,
                "solar_grid_charge": solar_grid_charge,
                "grid_charge": grid_charge,
                "discharge": discharge,
                "grid_export": grid_export,
                "grid_home": grid_home,
                "home_consumption": home_consumption,
                "state_of_charge": state_of_charge,
            }

            diffs = {key: None for key in current_values}
            if previous_state and isinstance(previous_state.get("values"), dict):
                previous_values = previous_state.get("values") or {}
                for key, value in current_values.items():
                    prev = previous_values.get(key)
                    if prev is None:
                        diffs[key] = None
                    else:
                        diffs[key] = float(value) - float(prev)

            metrics = AnkerMetrics(
                current_soc=current_soc,
                daily_counters=AnkerDailyCounters(
                    solar_charge=diffs.get("solar_charge"),
                    solar_yield=diffs.get("solar_yield"),
                    solar_grid_charge=diffs.get("solar_grid_charge"),
                    grid_charge=diffs.get("grid_charge"),
                    discharge=diffs.get("discharge"),
                    grid_export=diffs.get("grid_export"),
                    grid_home=diffs.get("grid_home"),
                    home_consumption=diffs.get("home_consumption"),
                    state_of_charge=diffs.get("state_of_charge"),
                    current_soc=current_soc,
                ),
            )
            self._save_state(current_values)
            return metrics

    def _state_path(self) -> Path:
        return Path(__file__).with_name(STATE_FILENAME)

    def _load_state(self) -> dict | None:
        path = self._state_path()
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _save_state(self, values: dict) -> None:
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "values": values,
        }
        path = self._state_path()
        path.write_text(json.dumps(payload), encoding="utf-8")

    def update_battery_actions(
        self,
        setpoint: float,
        tariff_group: str,
        usage_mode: str,
    ) -> None:
        """Update battery actions using the Anker API."""
        usage_map = {
            "unknown": SolarbankUsageMode.unknown,
            "smartmeter": SolarbankUsageMode.smartmeter,
            "smartplugs": SolarbankUsageMode.smartplugs,
            "manual": SolarbankUsageMode.manual,
            "backup": SolarbankUsageMode.backup,
            "use_time": SolarbankUsageMode.use_time,
            "smart": SolarbankUsageMode.smart,
            "time_slot": SolarbankUsageMode.time_slot,
        }
        tariff_map = {
            "peak": SolixTariffTypes.PEAK,
            "mid_peak": SolixTariffTypes.MID_PEAK,
            "off_peak": SolixTariffTypes.OFF_PEAK,
            "valley": SolixTariffTypes.VALLEY,
        }

        with Session() as websession:
            api = AnkerSolixApi(
                self.user,
                self.password,
                self.country,
                websession,
                self.logger,
            )
            api.authenticate()

            usage_value = usage_map[str(usage_mode).lower()]
            tariff_value = tariff_map[str(tariff_group).lower()]

            api.set_sb2_home_load(
                siteId=self.site_id,
                deviceSn=self.device_sn,
                preset=int(setpoint),
                usage_mode=int(usage_value),
            )

            if usage_value == SolarbankUsageMode.use_time:
                api.set_sb2_use_time(
                    siteId=self.site_id,
                    deviceSn=self.device_sn,
                    day_type="all",
                    start_hour=0,
                    end_hour=24,
                    tariff_type=int(tariff_value),
                )


@dataclass(frozen=True)
class AnkerDailyCounters:
    solar_charge: float | None
    solar_yield: float | None
    solar_grid_charge: float | None
    grid_charge: float | None
    discharge: float | None
    grid_export: float | None
    grid_home: float | None
    home_consumption: float | None
    state_of_charge: float | None
    current_soc: float | None


@dataclass(frozen=True)
class AnkerMetrics:
    current_soc: float
    daily_counters: AnkerDailyCounters
