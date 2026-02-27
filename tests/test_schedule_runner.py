import logging
import os
from pathlib import Path

import pytest

import datetime

from common.db_repository import DEFAULT_DB_ENV_VARS, DbRepository, get_db_connection_string
from Scheduler.schedule_runner import (
    DEFAULT_BATTERY_CAPACITY_KWH,
    DEFAULT_HORIZON_HOURS,
    DEFAULT_STEP_MINUTES,
    run_optimization,
    run_optimization_and_store,
)


def _load_env_local() -> None:
    env_path = Path(__file__).resolve().parents[1] / "env.local"
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and value and key not in os.environ:
                os.environ[key] = value


def test_run_optimization_smoke():
    _load_env_local()

    tide_model_path = os.environ.get("ENERGYMODEL_TIDE_PATH")
    solar_model_path = os.environ.get("ENERGYMODEL_SOLAR_PATH")
    if not tide_model_path or not solar_model_path:
        pytest.skip("ENERGYMODEL_TIDE_PATH and ENERGYMODEL_SOLAR_PATH must be set for this test")

    db_conn = os.environ.get("energydb") or os.environ.get("ENERGYDB")
    if not db_conn:
        pytest.skip("ENERGYDB/energydb must be set for this test")

    model, results, inputs = run_optimization(
        horizon=DEFAULT_HORIZON_HOURS,
        time_limit_sec=5,
        step_minutes=DEFAULT_STEP_MINUTES,
        now_utc=datetime.datetime.now(datetime.timezone.utc),
    )

    assert model is not None
    assert inputs is not None
    assert len(inputs.consumption_kwh) > 0
    assert len(inputs.solar_kwh) > 0
    assert len(inputs.price_per_kwh) > 0

    # Solver may be unavailable; in that case results will be None.
    if results is not None:
        assert results.solver is not None

        print("solver_status", getattr(results.solver, "status", None))
        print("termination_condition", getattr(results.solver, "termination_condition", None))
        print("objective", model.objective())

        for t in model.T:
            print(
                "t",
                int(t),
                "price",
                inputs.price_per_kwh[int(t)],
                "charge_on",
                model.charge_on[t](),
                "discharge_on",
                model.discharge_on[t](),
                "solar_charge_on",
                model.solar_charge_on[t](),
                "soc_kwh",
                model.soc_kwh[t](),
                "grid_kwh",
                model.grid_kwh[t](),
                "grid_cost",
                model.grid_kwh[t]() * inputs.price_per_kwh[int(t)],
            )


def test_run_optimization_and_store_smoke():
    _load_env_local()

    tide_model_path = os.environ.get("ENERGYMODEL_TIDE_PATH")
    solar_model_path = os.environ.get("ENERGYMODEL_SOLAR_PATH")
    if not tide_model_path or not solar_model_path:
        pytest.skip("ENERGYMODEL_TIDE_PATH and ENERGYMODEL_SOLAR_PATH must be set for this test")

    db_conn = os.environ.get("energydb") or os.environ.get("ENERGYDB")
    if not db_conn:
        pytest.skip("ENERGYDB/energydb must be set for this test")

    # Smoke test: should run without raising.
    try:
        run_optimization_and_store(
            horizon=DEFAULT_HORIZON_HOURS,
            time_limit_sec=5,
            step_minutes=DEFAULT_STEP_MINUTES,
        )
    except Exception as exc:
        print("test_run_optimization_and_store_smoke failed with:", repr(exc))
        raise


def test_print_current_soc():
    _load_env_local()

    db_conn = os.environ.get("energydb") or os.environ.get("ENERGYDB")
    if not db_conn:
        pytest.skip("ENERGYDB/energydb must be set for this test")

    connection_string = get_db_connection_string(DEFAULT_DB_ENV_VARS)
    with DbRepository(connection_string=connection_string, logger=logging.getLogger(__name__)) as repo:
        row = repo.get_current_battery_state()
        if not row:
            raise ValueError("No current battery state found")
        _, soc_value = row
        soc_kwh = (float(soc_value) / 100.0) * DEFAULT_BATTERY_CAPACITY_KWH

    print("current_soc_kwh", soc_kwh)
    assert isinstance(soc_kwh, float)
