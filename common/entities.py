from sqlalchemy import Column, Integer, String, ForeignKey, Date, Float, Table, DateTime, Text, text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Boolean

Base = declarative_base()

class WeatherRecord(Base):
    __tablename__ = 'Weather'
    timestamp = Column(DateTime,primary_key=True, nullable=False)
    pressure = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    cloud_are_fraction = Column(Float, nullable=False)
    precipitation = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    wind_from_direction = Column(Float, nullable=False)
    wind_speed = Column(Float, nullable=False)

class AnkerData(Base):
    __tablename__ = 'AnkerData'
    Timestamp = Column(DateTime, primary_key=True, nullable=False)
    battery_health = Column(Float, nullable=True)
    battery_diff = Column(Float, nullable=True)
    grid_export = Column(Float, nullable=True)
    grid_import = Column(Float, nullable=True)
    home_consumption = Column(Float, nullable=True)
    discharge = Column(Float, nullable=True)
    grid_charge = Column(Float, nullable=True)
    solar_total = Column(Float, nullable=True)
    solar_charge = Column(Float, nullable=True)

class DailyCounter(Base):
    __tablename__ = 'hm_daily_counter'
    bucket_ts = Column(DateTime, primary_key=True, nullable=False)
    entity_id = Column(String(255), primary_key=True, nullable=False)
    delta = Column(Float, nullable=True)
    current_state = Column(Float, nullable=True)
    delta_raw = Column(Float, nullable=True)

class EnergyPrice(Base):
    __tablename__ = 'ElectricityPrice'
    __table_args__ = {'schema': 'energy'}
    timestamp = Column('Timestamp', DateTime, nullable=False, primary_key=True)
    price = Column('price', Float, nullable=False)
    price_excl_tax = Column('price_excl_tax', Float, nullable=False)


class AnkerDataOld(Base):
    __tablename__ = 'AnkerData_old'
    Timestamp = Column(DateTime, primary_key=True, nullable=False)
    grid_to_battery_total = Column(Float, nullable=True)
    solar_to_battery_total = Column(Float, nullable=True)
    solar_to_grid_total = Column(Float, nullable=True)
    solar_to_home_total = Column(Float, nullable=True)
    battery_to_home_total = Column(Float, nullable=True)
    grid_to_home_total = Column(Float, nullable=True)
    battery_health = Column(Float, nullable=True)
    battery_diff = Column(Float, nullable=True)
    battery_discharge_total = Column(Float, nullable=True)

class CurrentState(Base):
    __tablename__ = 'hm_current_state'
    __table_args__ = {'schema': 'energy'}
    last_updated = Column('last_updated', DateTime, nullable=False, primary_key=True)
    value = Column('state', Float, nullable=False)
    name = Column('entity_id', String(255), nullable=False)


class Scheduler(Base):
    __tablename__ = 'scheduler'
    __table_args__ = {'schema': 'energy'}
    timestamp = Column(DateTime, nullable=False, primary_key=True)
    charge_on = Column(Boolean, nullable=True)
    solar_charge_on = Column(Boolean, nullable=True)
    discharge_on = Column(Boolean, nullable=True)
    solar = Column(Float, nullable=True)
    consumption = Column(Float, nullable=True)
    grid = Column(Float, nullable=True)
    battery = Column(Float, nullable=True)
