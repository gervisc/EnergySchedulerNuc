import datetime
import logging
import os
from typing import Optional, Tuple


import pyomo.environ as pyo
from darts.models import LinearRegressionModel, TiDEModel

from common.db_repository import DbRepository, DEFAULT_DB_ENV_VARS, get_db_connection_string
from common.time_features import DEFAULT_LATITUDE, DEFAULT_LOCAL_TZ_NAME, DEFAULT_LONGITUDE
from Scheduler.forecast_service import ForecastService
from Scheduler.homeassistant import update_battery_actions
from Scheduler.optimizer import OptimizationInputs, build_battery_milp, expand_inputs_to_steps
from pyomo.opt.results import SolverResults

LOGGER = logging.getLogger(__name__)
SOLVER_NAME = "glpk"
DEFAULT_HORIZON_HOURS = 24
DEFAULT_STEP_MINUTES = 15
DEFAULT_TIME_LIMIT_SEC: int = 30
DEFAULT_BATTERY_CAPACITY_KWH = float(os.environ.get("BATTERY_CAPACITY_KWH", "1.6"))
ENERGYMODEL_TIDE_ENV = "ENERGYMODEL_TIDE_PATH"
ENERGYMODEL_SOLAR_ENV = "ENERGYMODEL_SOLAR_PATH"
# Create a formatter for the log messages
LOGGER.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set the log level for the console handler
console_handler.setFormatter(formatter)  # Apply the formatter to the console handler

# Add both handlers to the logger
#logger.addHandler(file_handler)
LOGGER.addHandler(console_handler)

def _series_to_list(series, horizon: Optional[int]) -> list[float]:
    df = series.to_dataframe()
    column = df.columns[0]
    values = df[column].astype("float64").tolist()
    if horizon is not None:
        values = values[:horizon]
    return values


def build_inputs_from_db(
    repo: DbRepository,
    forecast_service: ForecastService,
    horizon: int,
    consumption_model: TiDEModel,
    solar_model: LinearRegressionModel,
) -> OptimizationInputs:
    consumption_ts = forecast_service.predict_next_24_hours_consumption(
        model=consumption_model,
        horizon=horizon,
    )
    solar_ts = forecast_service.predict_next_24_hours_solar(
        model=solar_model,
        horizon=horizon,
    )

    consumption_kwh = _series_to_list(consumption_ts, horizon=horizon)
    solar_kwh = _series_to_list(solar_ts, horizon=horizon)

    price_rows = repo.get_current_and_next_24_hours_prices()
    price_per_kwh = [price for _, price, _ in price_rows][:horizon]

    expected_sell = repo.get_expected_discharge_sell_price() or 0.0

    row = repo.get_current_battery_state()
    if not row:
        raise ValueError("No current battery state found")
    _, soc_value = row
    current_soc_kwh = (float(soc_value) / 100.0) * DEFAULT_BATTERY_CAPACITY_KWH

    horizon = min(len(consumption_kwh), len(solar_kwh), len(price_per_kwh))
    if horizon <= 0:
        raise ValueError("Not enough data to build optimization inputs")

    return OptimizationInputs(
        consumption_kwh=consumption_kwh[:horizon],
        solar_kwh=solar_kwh[:horizon],
        price_per_kwh=price_per_kwh[:horizon],
        expected_discharge_sell_price=expected_sell,
        current_soc_kwh=current_soc_kwh,
    )


def run_optimization(
    horizon: int,
    time_limit_sec: int,
    step_minutes: int,
    now_utc: datetime.datetime,
) -> Tuple[pyo.ConcreteModel, Optional[SolverResults], OptimizationInputs]:
    connection_string = get_db_connection_string(DEFAULT_DB_ENV_VARS)
    with DbRepository(connection_string=connection_string, logger=LOGGER) as repo:
        forecast_service = ForecastService(
            repo,
            local_tz_name=DEFAULT_LOCAL_TZ_NAME,
            longitude=DEFAULT_LONGITUDE,
            latitude=DEFAULT_LATITUDE,
        )
        tide_model_path = os.environ.get(ENERGYMODEL_TIDE_ENV)
        if not tide_model_path:
            raise ValueError(f"{ENERGYMODEL_TIDE_ENV} is not set; cannot load forecast model")
        solar_model_path = os.environ.get(ENERGYMODEL_SOLAR_ENV)
        if not solar_model_path:
            raise ValueError(f"{ENERGYMODEL_SOLAR_ENV} is not set; cannot load solar model")

        tide_model = forecast_service.load_model(tide_model_path)
        solar_model = forecast_service.load_solar_model(solar_model_path)
        inputs = build_inputs_from_db(
            repo,
            forecast_service,
            horizon=horizon,
            consumption_model=tide_model,
            solar_model=solar_model,
        )
        inputs = expand_inputs_to_steps(inputs, step_minutes, now_utc=now_utc)
        model = build_battery_milp(inputs, step_minutes=step_minutes)

        solver = pyo.SolverFactory(SOLVER_NAME)
        if solver is None or not solver.available():
            return model, None, inputs

        if time_limit_sec is not None:
            solver.options["tmlim"] = int(time_limit_sec)

        results = solver.solve(model, tee=False)
        return model, results, inputs


def run_optimization_and_store(
    horizon: int,
    time_limit_sec: int,
    step_minutes: int,
) -> None:
    usage_mode = "use_time"
    tariff_group = "off_peak"
    preset = 0.0
    try:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        model, results, inputs = run_optimization(
            horizon=horizon,
            time_limit_sec=time_limit_sec,
            step_minutes=step_minutes,
            now_utc=now_utc,
        )
        if results is None:
            return

        horizon_len = min(
            len(inputs.consumption_kwh),
            len(inputs.solar_kwh),
            len(inputs.price_per_kwh),
        )
        step_seconds = step_minutes * 60
        seconds_in_hour = (
            now_utc.minute * 60
            + now_utc.second
            + now_utc.microsecond / 1_000_000.0
        )
        remainder = seconds_in_hour % step_seconds
        if remainder < 1e-9:
            start = now_utc.replace(second=0, microsecond=0)
        else:
            delta = step_seconds - remainder
            start = now_utc + datetime.timedelta(seconds=delta)
            start = start.replace(second=0, microsecond=0)
        timestamps = [
            start + datetime.timedelta(minutes=step_minutes * i)
            for i in range(horizon_len)
        ]

        rows = []
        for t in model.T:
            idx = int(t)
            if idx >= horizon_len:
                break
            rows.append(
                (
                    timestamps[idx],
                    bool(round(model.charge_on[t]())),
                    bool(round(model.solar_charge_on[t]())),
                    bool(round(model.discharge_on[t]())),
                    float(inputs.solar_kwh[idx]),
                    float(inputs.consumption_kwh[idx]),
                    float(model.grid_kwh[t]()),
                    float(model.soc_kwh[t]()),
                )
            )

        connection_string = get_db_connection_string(DEFAULT_DB_ENV_VARS)
        with DbRepository(connection_string=connection_string, logger=LOGGER) as repo:
            repo.upsert_scheduler_rows(rows)

        first_idx = int(model.T.first())
        solar_charge_on = bool(round(model.solar_charge_on[first_idx]()))
        charge_on = bool(round(model.charge_on[first_idx]()))
        discharge_on = bool(round(model.discharge_on[first_idx]()))
        soc_kwh = float(model.soc_kwh[first_idx]())

        if solar_charge_on:
            usage_mode = "manual"
            preset = 0.0
        elif charge_on:
            usage_mode = "use_time"
            tariff_group = "valley"
        elif discharge_on:
            usage_mode = "manual"
            preset = (0.4 + float(inputs.solar_kwh[first_idx])) * 1000.0
        elif soc_kwh < 1.6 * 0.8:
            usage_mode = "use_time"
            tariff_group = "off_peak"
        else:
            usage_mode = "manual"
            preset = float(inputs.solar_kwh[first_idx]) * 1000.0


    except Exception as e:
        LOGGER.exception("Failed to run optimization and store results: %s", e)
    LOGGER.info(
        "Optimization result: usage_mode=%s tariff_group=%s preset=%.1f",
        usage_mode,
        tariff_group,
        preset,
    )
    update_battery_actions(preset, tariff_group, usage_mode)

    return

    
        




if __name__ == "__main__":
    run_optimization_and_store(
        horizon=DEFAULT_HORIZON_HOURS,
        time_limit_sec=DEFAULT_TIME_LIMIT_SEC,
        step_minutes=DEFAULT_STEP_MINUTES,
    )
