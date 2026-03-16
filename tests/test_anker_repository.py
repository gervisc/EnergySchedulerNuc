import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from requests import Session
from requests.exceptions import RequestException

from common import anker_repository as ar
from common.env_utils import load_repo_env_local


def _load_env_local() -> None:
    load_repo_env_local(Path(__file__))


def _repo_from_env() -> ar.AnkerRepository:
    _load_env_local()
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
        logger=logging.getLogger("test"),
    )


def _api_from_env() -> ar.AnkerSolixApi:
    _load_env_local()
    user = os.getenv("ANKERUSER")
    password = os.getenv("ANKERPASSWORD")
    country = os.getenv("ANKERCOUNTRY")
    if not all([user, password, country]):
        pytest.skip("ANKERUSER, ANKERPASSWORD, ANKERCOUNTRY must be set")
    websession = Session()
    api = ar.AnkerSolixApi(user, password, country, websession, logging.getLogger("test"))
    api.authenticate()
    return api


def test_get_current_metrics_integration():
    repo = _repo_from_env()
    metrics = repo.get_current_metrics(timestep_minutes=15)
    assert metrics is not None
    assert metrics.daily_counters is not None


def test_update_battery_actions_integration():
    _load_env_local()


    repo = _repo_from_env()
    repo.update_battery_actions(setpoint=400, tariff_group="valley", usage_mode="manual")


def test_update_battery_actions_manual_sets_mode_integration():
    repo = _repo_from_env()
    try:
        repo.update_battery_actions(setpoint=400, tariff_group="valley", usage_mode="manual")
    except RequestException as exc:
        pytest.skip(f"Skipping integration test due to API/network error: {exc}")

    _load_env_local()
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
