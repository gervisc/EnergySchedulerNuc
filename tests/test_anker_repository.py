import logging
import os
from pathlib import Path

import pytest

from common import anker_repository as ar


def _load_env_local() -> None:
     env_file =   Path(__file__).resolve().parents[1] / "env.local"
     if env_file.exists():
        with env_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                if key and value and key not in os.environ:
                    os.environ[key] = value


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


def test_get_current_metrics_integration():
    repo = _repo_from_env()
    metrics = repo.get_current_metrics(timestep_minutes=15)
    assert metrics is not None
    assert metrics.daily_counters is not None


def test_update_battery_actions_integration():
    _load_env_local()


    repo = _repo_from_env()
    repo.update_battery_actions(setpoint=0, tariff_group="valley", usage_mode="manual")
