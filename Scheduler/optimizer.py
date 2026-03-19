import datetime
import math
from dataclasses import dataclass
from typing import Sequence

import pyomo.environ as pyo

@dataclass
class OptimizationInputs:
    consumption_kwh: Sequence[float]
    solar_kwh: Sequence[float]
    price_per_kwh: Sequence[float]
    expected_discharge_sell_price: float
    current_soc_kwh: float


def clamp_near_zero(x: float, eps: float = 1e-6) -> float:
    return 0.0 if abs(x) < eps else x


def _remaining_steps_in_hour(now_utc: datetime.datetime, step_minutes: int) -> int:
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=datetime.timezone.utc)
    else:
        now_utc = now_utc.astimezone(datetime.timezone.utc)

    minutes = (
        now_utc.minute
        + now_utc.second / 60.0
        + now_utc.microsecond / 3_600_000_000.0
    )
    steps_per_hour = 60 // step_minutes
    remainder = minutes % step_minutes
    if remainder < 0.5*step_minutes:
        steps_elapsed = int(minutes // step_minutes)
    else:
        steps_elapsed = int(math.ceil(minutes / step_minutes))
    return max(steps_per_hour - steps_elapsed, 0)


def expand_inputs_to_steps(
    inputs: OptimizationInputs,
    step_minutes: int,
    now_utc: datetime.datetime,
) -> OptimizationInputs:
    if step_minutes <= 0 or 60 % step_minutes != 0:
        raise ValueError("step_minutes must be a positive divisor of 60")

    steps_per_hour = 60 // step_minutes
    first_steps = _remaining_steps_in_hour(now_utc, step_minutes)

    def _expand(values: Sequence[float]) -> list[float]:
        expanded: list[float] = []
        for idx, value in enumerate(values):
            repeats = first_steps if idx == 0 else steps_per_hour
            if repeats <= 0:
                continue
            expanded.extend([float(value)] * repeats)
        return expanded

    return OptimizationInputs(
        consumption_kwh=_expand(inputs.consumption_kwh),
        solar_kwh=_expand(inputs.solar_kwh),
        price_per_kwh=_expand(inputs.price_per_kwh),
        expected_discharge_sell_price=inputs.expected_discharge_sell_price,
        current_soc_kwh=inputs.current_soc_kwh,
    )


def build_battery_milp(
    inputs: OptimizationInputs,
    step_minutes: int,
) -> pyo.ConcreteModel:
    """Build a simple MILP for battery charging decisions.

    Decision: charge or not per hour (binary). Objective: minimize grid cost.
    """
    inputs = OptimizationInputs(
        consumption_kwh=[clamp_near_zero(v) for v in inputs.consumption_kwh],
        solar_kwh=[clamp_near_zero(v) for v in inputs.solar_kwh],
        price_per_kwh=[clamp_near_zero(v) for v in inputs.price_per_kwh],
        expected_discharge_sell_price=clamp_near_zero(inputs.expected_discharge_sell_price),
        current_soc_kwh=clamp_near_zero(inputs.current_soc_kwh),
    )

    horizon = min(len(inputs.consumption_kwh), len(inputs.solar_kwh), len(inputs.price_per_kwh))
    if horizon <= 0:
        raise ValueError("No horizon available for optimization")

    dt_hours = [step_minutes / 60.0] * horizon

    m = pyo.ConcreteModel(name="battery_schedule")
    m.T = pyo.RangeSet(0, horizon - 1)

    m.charge_on = pyo.Var(m.T, within=pyo.Binary)
    m.discharge_on = pyo.Var(m.T, within=pyo.Binary)
    m.solar_charge_on = pyo.Var(m.T, within=pyo.Binary)
    m.soc_kwh = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.grid_kwh = pyo.Var(m.T, within=pyo.Reals)

    
     
    battery_loss = 0.018 #kWh lost per hour when battery is active (charging or discharging)
    charge_efficiency = 0.8
    discharge_rate =0.400
    charge_rate = 0.400
    solar_charge_efficiency = 0.85
    battery_capcity_kwh = 1.6
    minimum_level = 0.2 * battery_capcity_kwh
    if(inputs.current_soc_kwh < minimum_level):
        inputs.current_soc_kwh = minimum_level

    def _loss_term(t: int) -> float:
        return battery_loss * dt_hours[t] if inputs.solar_kwh[t] <= 0 else 0.0

    def _charge_rate_for_step(t: int) -> float:
        max_charge_rate = (
            (battery_capcity_kwh - inputs.current_soc_kwh)
            / (charge_efficiency * dt_hours[t])
        )
        if t == 0 and max_charge_rate < charge_rate:
            rate = max(math.floor(max_charge_rate * 1000.0) / 1000.0, 0.0)
        else:
            rate = charge_rate
        return rate

    def _discharge_rate_for_step(t: int) -> float:
        max_discharge_rate = (
            (inputs.current_soc_kwh - minimum_level - _loss_term(t))
            * charge_efficiency
            / dt_hours[t]
        )
        if t == 0 and max_discharge_rate < discharge_rate:
            rate = max(math.floor(max_discharge_rate * 1000.0) / 1000.0, 0.0)
        else:
            rate = discharge_rate
        return rate

    #battery balance
    def battery_balance_rule(m, t):
            prev_energy = (inputs.current_soc_kwh if t==0 else m.soc_kwh[t - 1])
            step_charge_rate = _charge_rate_for_step(int(t))
            step_discharge_rate = _discharge_rate_for_step(int(t))
            return (
                m.soc_kwh[t] ==  prev_energy +
                m.solar_charge_on[t] * inputs.solar_kwh[t] * dt_hours[t] * solar_charge_efficiency
                - m.discharge_on[t] * step_discharge_rate / charge_efficiency * dt_hours[t]
                + m.charge_on[t] * step_charge_rate * charge_efficiency * dt_hours[t]
                - _loss_term(t)
        )

    m.soc_update = pyo.Constraint(m.T, rule=battery_balance_rule)

   


    # Constraint definition: grid import (no-export vs export-allowed).
    def grid_rule(m, t):
        step_charge_rate = _charge_rate_for_step(int(t))
        step_discharge_rate = _discharge_rate_for_step(int(t))
        return (m.grid_kwh[t] == 
        - (1-m.solar_charge_on[t]) * inputs.solar_kwh[t]* dt_hours[t] 
        - (m.discharge_on[t]) * step_discharge_rate * dt_hours[t] 
        + (m.charge_on[t]) * step_charge_rate * dt_hours[t]
        + inputs.consumption_kwh[t] * dt_hours[t]
        )
    m.grid = pyo.Constraint(m.T, rule=grid_rule)
   
   
   
    # Constraint definition: SOC upper bound.
    def soc_cap_rule(m, t):
        return (minimum_level, m.soc_kwh[t], battery_capcity_kwh)


    m.soc_cap = pyo.Constraint(m.T, rule=soc_cap_rule)

    def one_mode_rule(m, t):
        return m.discharge_on[t] + m.charge_on[t] + m.solar_charge_on[t] <= 1

    m.one_mode = pyo.Constraint(m.T, rule=one_mode_rule)





    last_t = m.T.last()

    def objective_rule(m):
        total_cost = 0.0
        for t in m.T:
            total_cost += (
                inputs.price_per_kwh[t] * m.grid_kwh[t]
                - (m.soc_kwh[t] * inputs.expected_discharge_sell_price * charge_efficiency if t == last_t else 0)
            )
        return total_cost

    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return m
