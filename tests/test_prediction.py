import logging
import os
from pathlib import Path

import pandas as pd
import pytest

from common.db_repository import DEFAULT_DB_ENV_VAR, DbRepository, get_db_connection_string
from common.env_utils import load_repo_env_local
from common.time_features import DEFAULT_LATITUDE, DEFAULT_LOCAL_TZ_NAME, DEFAULT_LONGITUDE
from Scheduler.forecast_service import ForecastService


def test_predict_next_24_hours_consumption_smoke():
    load_repo_env_local(Path(__file__))

    model_path = os.environ.get("ENERGYMODEL_TIDE_PATH")
    if not model_path:
        pytest.skip("ENERGYMODEL_TIDE_PATH must be set for this test")

    db_conn = os.environ.get("energydb") or os.environ.get("ENERGYDB")
    if not db_conn:
        pytest.skip("ENERGYDB/energydb must be set for this test")

    from darts.models import TiDEModel
    if not callable(getattr(TiDEModel, "load", None)):
        pytest.skip("TiDEModel is unavailable; install Darts with required extras")

    connection_string = get_db_connection_string(DEFAULT_DB_ENV_VAR)
    repo = DbRepository(connection_string=connection_string, logger=logging.getLogger(__name__))
    try:
        forecast_service = ForecastService(
            repo,
            local_tz_name=DEFAULT_LOCAL_TZ_NAME,
            longitude=DEFAULT_LONGITUDE,
            latitude=DEFAULT_LATITUDE,
        )
        model = forecast_service.load_model(model_path)
        prediction = forecast_service.predict_next_24_hours_consumption(model=model, horizon=24)
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
    load_repo_env_local(Path(__file__))

    model_path = os.environ.get("ENERGYMODEL_SOLAR_PATH")
    if not model_path:
        pytest.skip("ENERGYMODEL_SOLAR_PATH must be set for this test")
    path = Path(model_path)
    if not path.exists():
        pytest.skip("ENERGYMODEL_SOLAR_PATH does not exist; solar model not available")

    db_conn = os.environ.get("energydb") or os.environ.get("ENERGYDB")
    if not db_conn:
        pytest.skip("ENERGYDB/energydb must be set for this test")

    connection_string = get_db_connection_string(DEFAULT_DB_ENV_VAR)
    repo = DbRepository(connection_string=connection_string, logger=logging.getLogger(__name__))
    try:
        if not repo.get_current_and_next_24_hours_weather():
            pytest.skip("No weather data available for the next 24 hours")

        forecast_service = ForecastService(
            repo,
            local_tz_name=DEFAULT_LOCAL_TZ_NAME,
            longitude=DEFAULT_LONGITUDE,
            latitude=DEFAULT_LATITUDE,
        )
        model = forecast_service.load_solar_model(str(path))
        prediction = forecast_service.predict_next_24_hours_solar(model=model, horizon=24)
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
