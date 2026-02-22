import os
from pathlib import Path

import pandas as pd
import pytest

from common.db_repository import DbRepository
from Scheduler.forecast_service import ForecastService


def test_predict_next_24_hours_consumption_smoke():
    env_path = Path(__file__).resolve().parents[1] / "env.local"
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")  # trim quotes
                if key and value and key not in os.environ:
                    os.environ[key] = value

    model_path = os.environ.get("ENERGYMODEL_TIDE_PATH")
    if not model_path:
        pytest.skip("ENERGYMODEL_TIDE_PATH must be set for this test")

    db_conn = os.environ.get("energydb") or os.environ.get("ENERGYDB")
    if not db_conn:
        pytest.skip("ENERGYDB/energydb must be set for this test")

    from darts.models import TiDEModel
    if not callable(getattr(TiDEModel, "load", None)):
        pytest.skip("TiDEModel is unavailable; install Darts with required extras")

    repo = DbRepository()
    try:
        forecast_service = ForecastService(repo)
        prediction = forecast_service.predict_next_24_hours_consumption()
        df = prediction.to_dataframe()
        print(df.head(24))

        # Basic sanity checks
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 24
    finally:
        try:
            repo.close()
        except Exception:
            # Ignore errors during cleanup if connection is already lost
            pass


def test_predict_next_24_hours_solar_smoke():
    env_path = Path(__file__).resolve().parents[1] / "env.local"
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")  # trim quotes
                if key and value and key not in os.environ:
                    os.environ[key] = value

    model_path = os.environ.get("ENERGYMODEL_SOLAR_PATH")
    if not model_path:
        pytest.skip("ENERGYMODEL_SOLAR_PATH must be set for this test")
    path = Path(model_path)
    if path.is_dir():
        path = path / "solar_yield_model"
    if not path.exists():
        pytest.skip("ENERGYMODEL_SOLAR_PATH does not exist; solar model not available")

    db_conn = os.environ.get("energydb") or os.environ.get("ENERGYDB")
    if not db_conn:
        pytest.skip("ENERGYDB/energydb must be set for this test")

    from darts.models import LinearRegressionModel
    if not callable(getattr(LinearRegressionModel, "load", None)):
        pytest.skip("LinearRegressionModel is unavailable; install Darts with required extras")

    repo = DbRepository()
    try:
        if not repo.get_current_and_next_24_hours_weather():
            pytest.skip("No weather data available for the next 24 hours")

        forecast_service = ForecastService(repo)
        prediction = forecast_service.predict_next_24_hours_solar()
        df = prediction.to_dataframe()
        print(df.head(24))

        # Basic sanity checks
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 24
    finally:
        try:
            repo.close()
        except Exception:
            # Ignore errors during cleanup if connection is already lost
            pass
