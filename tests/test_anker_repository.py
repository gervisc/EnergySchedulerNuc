import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from requests import Session
from requests.exceptions import RequestException

from common import anker_repository as ar
from common.env_utils import load_repo_env_local

LOGGER = logging.getLogger("test")


def _configured_log_level() -> int:
    level_name = os.getenv("LOG_LEVEL")
    if not level_name:
        raise ValueError("LOG_LEVEL must be set in the environment")
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid LOG_LEVEL: {level_name}")
    return level


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables from env.local file if it exists."""
    load_repo_env_local(Path(__file__), overwrite=True)
    log_level = _configured_log_level()
    LOGGER.setLevel(log_level)
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        LOGGER.addHandler(handler)
    for handler in LOGGER.handlers:
        handler.setLevel(log_level)


def _repo_from_env() -> ar.AnkerRepository:
    user = os.getenv("ANKERUSER")
    password = os.getenv("ANKERPASSWORD")
    country = os.getenv("ANKERCOUNTRY")
    site_id = os.getenv("SITE_ID")
    device_sn = os.getenv("DEVICE_SN")
    if not all([user, password, country, site_id, device_sn]):
        pytest.skip("ANKERUSER, ANKERPASSWORD, ANKERCOUNTRY, SITE_ID, DEVICE_SN must be set")
    return ar.AnkerRepository(
        user=user,
        password=password,
        country=country,
        site_id=site_id,
        device_sn=device_sn,
        logger=LOGGER,
    )


def _api_from_env() -> ar.AnkerSolixApi:
    user = os.getenv("ANKERUSER")
    password = os.getenv("ANKERPASSWORD")
    country = os.getenv("ANKERCOUNTRY")
    if not all([user, password, country]):
        pytest.skip("ANKERUSER, ANKERPASSWORD, ANKERCOUNTRY must be set")
    websession = Session()
    api = ar.AnkerSolixApi(user, password, country, websession, LOGGER)
    api.authenticate()
    return api


def test_get_current_metrics_integration():
    repo = _repo_from_env()
    metrics = repo.get_current_metrics(timestep_minutes=15)
    assert metrics is not None
    assert metrics.daily_counters is not None


def test_update_battery_actions_integration():
    repo = _repo_from_env()
    repo.update_battery_actions(setpoint=400, tariff_group="valley", usage_mode="manual")


def test_update_battery_actions_manual_sets_mode_integration():
    repo = _repo_from_env()
    try:
        repo.update_battery_actions(setpoint=400, tariff_group="valley", usage_mode="manual")
    except RequestException as exc:
        pytest.skip(f"Skipping integration test due to API/network error: {exc}")

    site_id = os.getenv("SITE_ID")
    device_sn = os.getenv("DEVICE_SN")
    if not all([site_id, device_sn]):
        pytest.skip("SITE_ID and DEVICE_SN must be set")

    api = _api_from_env()
    try:
        try:
            api.update_sites(siteId=site_id)
            api.update_site_details()
            api.update_device_details()
        except RequestException as exc:
            pytest.skip(f"Skipping integration test due to API/network error: {exc}")
        device = api.devices.get(device_sn) or {}
        assert int(device.get("preset_usage_mode")) == int(ar.SolarbankUsageMode.manual)
    finally:
        api.apisession.close()


def test_get_current_soc_from_mqtt_integration(monkeypatch: pytest.MonkeyPatch):
    repo = _repo_from_env()

    try:
        soc = repo._get_current_soc_from_mqtt()
    except RequestException as exc:
        pytest.skip(f"Skipping MQTT integration test due to API/network error: {exc}")
    except RuntimeError as exc:
        pytest.skip(f"Skipping MQTT integration test: {exc}")

    if soc is None:
        pytest.skip(
            "MQTT SOC could not be retrieved. This can happen if paho-mqtt is not installed, "
            "the device/account does not expose MQTT data, or MQTT access failed."
        )

    assert isinstance(soc, float)
    assert 0.0 <= soc <= 100.0


def test_update_battery_actions_manual_reapplies_usage_mode_mock(monkeypatch: pytest.MonkeyPatch):
    mock_api = MagicMock()

    class _FakeSession:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(ar, "Session", _FakeSession)
    monkeypatch.setattr(ar, "AnkerSolixApi", lambda *args, **kwargs: mock_api)

    repo = ar.AnkerRepository(
        user="user",
        password="pass",
        country="NL",
        site_id="site",
        device_sn="device",
        logger=logging.getLogger("test"),
    )

    repo.update_battery_actions(setpoint=400, tariff_group="valley", usage_mode="manual")

    assert mock_api.set_sb2_home_load.call_count == 2
    first_call = mock_api.set_sb2_home_load.call_args_list[0]
    second_call = mock_api.set_sb2_home_load.call_args_list[1]
    assert first_call.kwargs["usage_mode"] == int(ar.SolarbankUsageMode.manual)
    assert "set_slot" in first_call.kwargs
    assert second_call.kwargs == {
        "siteId": "site",
        "deviceSn": "device",
        "usage_mode": int(ar.SolarbankUsageMode.manual),
    }


def test_extract_soc_from_mqtt_payload_numeric():
    repo = ar.AnkerRepository(
        user="user",
        password="pass",
        country="NL",
        site_id="site",
        device_sn="device",
        logger=logging.getLogger("test"),
    )

    scene_data = {"solarbank_info": {"solarbank_list": [{"battery_power": "47.5"}]}}
    assert repo._get_current_soc_from_scene_data(scene_data) == 47.5


def test_get_current_soc_prefers_mqtt(monkeypatch: pytest.MonkeyPatch):
    repo = ar.AnkerRepository(
        user="user",
        password="pass",
        country="NL",
        site_id="site",
        device_sn="device",
        logger=logging.getLogger("test"),
    )

    monkeypatch.setattr(repo, "_get_current_soc_from_mqtt", lambda api=None: 54.0)

    scene_data = {"solarbank_info": {"solarbank_list": [{"battery_power": "12"}]}}
    assert repo._get_current_soc(api=MagicMock(), scene_data=scene_data) == 54.0


def test_get_current_soc_falls_back_to_scene_when_mqtt_unavailable(monkeypatch: pytest.MonkeyPatch):
    repo = ar.AnkerRepository(
        user="user",
        password="pass",
        country="NL",
        site_id="site",
        device_sn="device",
        logger=logging.getLogger("test"),
    )

    monkeypatch.setattr(repo, "_get_current_soc_from_mqtt", lambda api=None: None)

    scene_data = {"solarbank_info": {"solarbank_list": [{"battery_power": "23"}]}}
    assert repo._get_current_soc(api=MagicMock(), scene_data=scene_data) == 23.0


def test_get_current_soc_from_mqtt_returns_none_when_device_not_found(
    monkeypatch: pytest.MonkeyPatch,
):
    repo = ar.AnkerRepository(
        user="user",
        password="pass",
        country="NL",
        site_id="site",
        device_sn="device",
        logger=logging.getLogger("test"),
    )

    class _FakeSession:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeApi:
        def __init__(self, *args, **kwargs):
            self.apisession = MagicMock()
            self.devices = {}

        def authenticate(self):
            return True

        def update_sites(self, siteId):
            return None

    monkeypatch.setattr(ar, "Session", _FakeSession)
    monkeypatch.setattr(ar, "AnkerSolixApi", _FakeApi)

    assert repo._get_current_soc_from_mqtt() is None
