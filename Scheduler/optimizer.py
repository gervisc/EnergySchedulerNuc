import datetime
from dataclasses import dataclass
from typing import Sequence

import pyomo.environ as pyo
from pyomo.opt.results import SolverResults
from typing import Optional, Tuple

@dataclass
class OptimizationInputs:
    consumption_kwh: Sequence[float]
    solar_kwh: Sequence[float]
    price_per_kwh: Sequence[float]
    expected_discharge_sell_price: float
    current_soc_kwh: float


def _hour_fractions(now_utc: datetime.datetime, horizon: int) -> list[float]:
    """Return length (in hours) for each optimization step.

    The first step is shortened to the remaining fraction of the current hour.
    """
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=datetime.timezone.utc)
    else:
        now_utc = now_utc.astimezone(datetime.timezone.utc)

    minutes = now_utc.minute + now_utc.second / 60.0 + now_utc.microsecond / 3_600_000_000.0
    first = max(1.0 - minutes / 60.0, 1e-6)
    return [first] + [1.0] * (horizon - 1)


def build_battery_milp(
    inputs: OptimizationInputs,
    now_utc: datetime.datetime | None = None,
) -> pyo.ConcreteModel:
    """Build a simple MILP for battery charging decisions.

    Decision: charge or not per hour (binary). Objective: minimize grid cost.
    """
    horizon = min(len(inputs.consumption_kwh), len(inputs.solar_kwh), len(inputs.price_per_kwh))
    if horizon <= 0:
        raise ValueError("No horizon available for optimization")

    if now_utc is None:
        now_utc = datetime.datetime.now(datetime.timezone.utc)

    dt_hours = _hour_fractions(now_utc, horizon)

    m = pyo.ConcreteModel(name="battery_schedule")
    m.T = pyo.RangeSet(0, horizon - 1)

    m.charge_on = pyo.Var(m.T, within=pyo.Binary)
    m.discharge_on = pyo.Var(m.T, within=pyo.Binary)
    m.solar_charge_on = pyo.Var(m.T, within=pyo.Binary)
    m.soc_kwh = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.grid_kwh = pyo.Var(m.T, within=pyo.Reals)

    
     
    battery_loss = 0.003 #kWh lost per hour when battery is active (charging or discharging)
    charge_efficiency = 0.8
    discharge_rate =0.400
    charge_rate = 0.400
    battery_capcity_kwh = 1.6
    minimum_level = 0.2 * battery_capcity_kwh


    #battery balance
    def battery_balance_rule(m, t):
            prev_energy = (inputs.current_soc_kwh if t==0 else m.soc_kwh[t - 1])
            return (
                m.soc_kwh[t] ==  prev_energy + 
                m.solar_charge_on[t] * inputs.solar_kwh[t] * dt_hours[t] 
                - m.discharge_on[t] * discharge_rate / charge_efficiency * dt_hours[t]
                + m.charge_on[t] * charge_rate * charge_efficiency * dt_hours[t]                   
                -battery_loss* dt_hours[t]
        )

    m.soc_update = pyo.Constraint(m.T, rule=battery_balance_rule)

   


    # Constraint definition: grid import (no-export vs export-allowed).
    def grid_rule(m, t):
        return (m.grid_kwh[t] == 
        - (1-m.solar_charge_on[t]) * inputs.solar_kwh[t]* dt_hours[t] 
        - (m.discharge_on[t]) * discharge_rate * dt_hours[t] 
        + (m.charge_on[t]) * charge_rate * dt_hours[t]
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
                + (m.soc_kwh[t] * inputs.expected_discharge_sell_price * charge_efficiency if t == 0 else 0)
                - (m.soc_kwh[t] * inputs.expected_discharge_sell_price * charge_efficiency if t == last_t else 0)
            )
        return total_cost

    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return m
