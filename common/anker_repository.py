from dataclasses import dataclass
from datetime import datetime, timedelta
import contextlib
import json
import logging
import os
from pathlib import Path

from requests import Session

from common.anker_api.api import AnkerSolixApi
from common.anker_api.mqtt import AnkerSolixMqttSession
from common.anker_api.apitypes import (
    Solarbank2Timeslot,
    SolarbankUsageMode,
    SolixDayTypes,
    SolixTariffTypes,
)

STATE_FILENAME = "anker_metrics_state.json"
API_CONTEXT_CACHE_FILENAME = "anker_api_context_cache.json"
API_CONTEXT_CACHE_TTL_SECONDS = 86400
USE_MQTT_FOR_CURRENT_SOC = True
MQTT_TIMEOUT_SECONDS = 10.0
MQTT_TRIGGER_SECONDS = 300


def _configured_log_level() -> int:
    level_name = os.environ.get("LOG_LEVEL")
    if not level_name:
        raise ValueError("LOG_LEVEL must be set in the environment")
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid LOG_LEVEL: {level_name}")
    return level


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
        self._websession: Session | None = None
        self._api: AnkerSolixApi | None = None
        configured_level = _configured_log_level()
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(configured_level)
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)
        self.logger.setLevel(configured_level)
        for handler in self.logger.handlers:
            handler.setLevel(configured_level)

    def close(self) -> None:
        if self._api is not None:
            with contextlib.suppress(Exception):
                self._api.apisession.close()
        if self._websession is not None:
            with contextlib.suppress(Exception):
                self._websession.close()
        self._api = None
        self._websession = None

    def __del__(self) -> None:
        self.close()

    def _get_api(self) -> AnkerSolixApi:
        if self._api is None:
            self._websession = Session()
            self._api = AnkerSolixApi(
                self.user,
                self.password,
                self.country,
                self._websession,
                self.logger,
            )
            self._api.authenticate()
        return self._api

    def get_current_metrics(self, timestep_minutes: int) -> "AnkerMetrics":
        """Return current SOC and diffs for today's energy counters."""
        now_local = datetime.now().astimezone()
        midnight_reset_window = (
            now_local.hour == 0 and now_local.minute < timestep_minutes
        )
        saved_ts = None
        saved_ts_local = None
        previous_state = self._load_state()
        if previous_state:
            timestamp = previous_state.get("timestamp")
            if timestamp:
                try:
                    saved_ts = datetime.fromisoformat(str(timestamp))
                except ValueError:
                    saved_ts = None
                if saved_ts is not None:
                    saved_ts_local = saved_ts.replace(tzinfo=now_local.tzinfo).astimezone(now_local.tzinfo)
                    age = now_local - saved_ts_local
                    if age < timedelta(minutes=timestep_minutes * 0.5):
                        raise ValueError(
                            "No previous Anker metrics state found; cannot calculate diffs yet."
                        )

        api = self._get_api()
        today = datetime.today()
        target_day = today - timedelta(days=1) if midnight_reset_window else today

        solar_data = api.energy_analysis(
            siteId=self.site_id,
            deviceSn=self.device_sn,
            startDay=target_day,
            endDay=target_day,
            rangeType="day",
            devType="solar_production",
        )

        home_data = api.energy_analysis(
            siteId=self.site_id,
            deviceSn=self.device_sn,
            startDay=target_day,
            endDay=target_day,
            rangeType="day",
            devType="home_usage",
        )
        solarbank_data = api.energy_analysis(
            siteId=self.site_id,
            deviceSn=self.device_sn,
            startDay=target_day,
            endDay=target_day,
            rangeType="day",
            devType="solarbank",
        )
        grid_data = api.energy_analysis(
            siteId=self.site_id,
            deviceSn=self.device_sn,
            startDay=target_day,
            endDay=target_day,
            rangeType="day",
            devType="grid",
        )

        def _to_float(value: object, default: float = 0.0) -> float:
            try:
                return float(value) if value is not None else default
            except (TypeError, ValueError):
                return default

        scene_data = api.get_scene_info(siteId=self.site_id)
        current_soc = self._get_current_soc(api=api, scene_data=scene_data)
        solar_charge = _to_float(solar_data["solar_to_battery_total"])
        solar_yield = _to_float(solar_data["solar_total"])
        solar_grid_charge = _to_float(solar_data["solar_to_grid_total"])
        grid_charge = _to_float(solarbank_data["charge_total"])
        discharge = _to_float(solarbank_data["battery_discharging_total"])
        grid_export = _to_float(grid_data["solar_to_grid_total"])
        grid_import = _to_float(home_data["grid_to_home_total"])
        home_consumption = _to_float(home_data["home_usage_total"])
        state_of_charge = current_soc

        current_values = {
            "solar_charge": solar_charge,
            "solar_yield": solar_yield,
            "solar_grid_charge": solar_grid_charge,
            "grid_charge": grid_charge,
            "discharge": discharge,
            "grid_export": grid_export,
            "grid_import": grid_import,
            "home_consumption": home_consumption,
            "state_of_charge": state_of_charge,
        }

        diffs = {key: None for key in current_values}
        if previous_state and isinstance(previous_state.get("values"), dict):
            previous_values = previous_state.get("values") or {}
            if saved_ts is None:
                diffs = {key: None for key in current_values}
            else:
                too_old = (now_local - saved_ts_local) > timedelta(minutes=timestep_minutes * 2)
                if too_old:
                    diffs = {key: None for key in current_values}
                else:
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
                grid_import=diffs.get("grid_import"),
                home_consumption=diffs.get("home_consumption"),
                state_of_charge=diffs.get("state_of_charge"),
                # current_soc is always the live SOC, never diffed or reset.
                current_soc=current_soc,
            ),
        )
        if midnight_reset_window:
            self._save_state({key: 0.0 for key in current_values})
        else:
            self._save_state(current_values)
        return metrics

    def _get_current_soc(self, api: AnkerSolixApi, scene_data: dict | None = None) -> float:
        mqtt_soc = self._get_current_soc_from_mqtt()
        if mqtt_soc is not None:
            return mqtt_soc

        fallback_soc = self._get_current_soc_from_scene_data(
            scene_data or api.get_scene_info(siteId=self.site_id)
        )
        self.logger.info(
            "Falling back to Anker scene/API current SOC: %s",
            fallback_soc,
        )
        return fallback_soc

    def _get_current_soc_from_scene_data(self, scene_data: dict) -> float:
        try:
            value = scene_data["solarbank_info"]["solarbank_list"][0]["battery_power"]
            return float(value)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("Unable to extract current SOC from Anker scene data") from exc

    def _get_current_soc_from_mqtt(self) -> float | None:
        if not USE_MQTT_FOR_CURRENT_SOC:
            return None
        timeout = MQTT_TIMEOUT_SECONDS
        trigger_timeout = MQTT_TRIGGER_SECONDS

        api = self._get_api()
        api.update_sites(siteId=self.site_id)
        return self._request_current_soc_via_mqtt(
            api=api,
            timeout=timeout,
            trigger_timeout=trigger_timeout,
        )

    def _request_current_soc_via_mqtt(
        self,
        api: AnkerSolixApi,
        timeout: float,
        trigger_timeout: int,
    ) -> float | None:
        device = api.devices.get(self.device_sn) or {}
        if not device:
            self.logger.warning(
                "Device %s not found in Anker API device cache; falling back to API current SOC",
                self.device_sn,
            )
            return None

        try:
            mqtt_session = AnkerSolixMqttSession(
                apisession=api.apisession,
                logger=self.logger,
            )
            return mqtt_session.request_device_soc(
                device_dict=device,
                timeout_seconds=timeout,
                realtime_trigger_seconds=trigger_timeout,
            )
        except Exception:
            self.logger.warning("Failed to read current SOC from Anker MQTT", exc_info=True)
            return None

    def _state_path(self) -> Path:
        return Path(__file__).with_name(STATE_FILENAME)

    def _load_state(self) -> dict | None:
        path = self._state_path()
        if not path.exists():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            values = state.get("values")
            if isinstance(values, dict) and "grid_import" not in values and "grid_home" in values:
                values["grid_import"] = values.pop("grid_home")
            return state
        except (OSError, json.JSONDecodeError):
            return None

    def _save_state(self, values: dict) -> None:
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "values": values,
        }
        path = self._state_path()
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _api_context_cache_path(self) -> Path:
        return Path(__file__).with_name(API_CONTEXT_CACHE_FILENAME)

    def _load_api_context_cache(self, max_age_seconds: int) -> dict | None:
        path = self._api_context_cache_path()
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            if payload.get("site_id") != self.site_id or payload.get("device_sn") != self.device_sn:
                return None
            timestamp_raw = payload.get("timestamp")
            if not timestamp_raw:
                return None
            timestamp = datetime.fromisoformat(str(timestamp_raw))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=datetime.now().astimezone().tzinfo)
            age = datetime.now().astimezone() - timestamp.astimezone()
            if age.total_seconds() > float(max_age_seconds):
                return None
            if not isinstance(payload.get("sites"), dict) or not isinstance(payload.get("devices"), dict):
                return None
            return payload
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            return None

    def _save_api_context_cache(self, api: AnkerSolixApi) -> None:
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "site_id": self.site_id,
            "device_sn": self.device_sn,
            "sites": api.sites,
            "devices": api.devices,
        }
        path = self._api_context_cache_path()
        try:
            path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        except (OSError, TypeError):
            self.logger.debug("Failed to persist API context cache", exc_info=True)

    def _prepare_api_context(self, api: AnkerSolixApi) -> None:
        max_age_seconds = int(
            os.environ.get(
                "ANKER_API_CONTEXT_CACHE_TTL_SECONDS",
                str(API_CONTEXT_CACHE_TTL_SECONDS),
            )
        )
        cached = self._load_api_context_cache(max_age_seconds=max_age_seconds)
        if cached:
            api.sites = cached["sites"]
            api.devices = cached["devices"]

        options = api.solarbank_usage_mode_options(deviceSn=self.device_sn)
        if options:
            return

        api.update_sites(siteId=self.site_id)
        api.update_site_details()
        api.update_device_details()
        self._save_api_context_cache(api)

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

        api = self._get_api()
        # Load device/site context from file cache when possible and refresh only if needed.
        self._prepare_api_context(api)

        usage_value = usage_map[str(usage_mode).lower()]
        tariff_value = tariff_map[str(tariff_group).lower()]

        if usage_value == SolarbankUsageMode.manual:
            api.set_sb2_home_load(
                siteId=self.site_id,
                deviceSn=self.device_sn,
                usage_mode=int(usage_value),
                set_slot=Solarbank2Timeslot(
                    start_time=datetime.strptime("00:00", "%H:%M"),
                    end_time=datetime.strptime("23:59", "%H:%M"),
                    appliance_load=int(setpoint),
                    weekdays=set(range(7)),
                ),
            )
            # Re-apply usage mode to ensure manual mode becomes active immediately.
            api.set_sb2_home_load(
                siteId=self.site_id,
                deviceSn=self.device_sn,
                usage_mode=int(usage_value),
            )
        else:
            if usage_value == SolarbankUsageMode.use_time:
                if tariff_value == SolixTariffTypes.VALLEY:
                    api.set_sb2_use_time(
                        siteId=self.site_id,
                        deviceSn=self.device_sn,
                        start_month=1,
                        end_month=12,
                        start_hour=0,
                        end_hour=24,
                        day_type=SolixDayTypes.ALL,
                        tariff_type=int(tariff_value),
                    )
                else:
                    api.set_sb2_use_time(
                        siteId=self.site_id,
                        deviceSn=self.device_sn,
                        day_type=SolixDayTypes.ALL,
                        start_hour=0,
                        end_hour=24,
                        tariff_type=int(tariff_value),
                    )
                # Re-apply usage mode to ensure use_time becomes active immediately.
                api.set_sb2_home_load(
                    siteId=self.site_id,
                    deviceSn=self.device_sn,
                    usage_mode=int(usage_value),
                )
            else:
                api.set_sb2_home_load(
                    siteId=self.site_id,
                    deviceSn=self.device_sn,
                    preset=int(setpoint),
                    usage_mode=int(usage_value),
                )

@dataclass(frozen=True)
class AnkerDailyCounters:
    solar_charge: float | None
    solar_yield: float | None
    solar_grid_charge: float | None
    grid_charge: float | None
    discharge: float | None
    grid_export: float | None
    grid_import: float | None
    home_consumption: float | None
    state_of_charge: float | None
    current_soc: float | None


@dataclass(frozen=True)
class AnkerMetrics:
    current_soc: float
    daily_counters: AnkerDailyCounters
