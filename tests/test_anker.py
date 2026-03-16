#!/usr/bin/env python
"""Pytest tests for the Anker API."""

from datetime import datetime, timedelta
import logging
import os
from pathlib import Path

import pytest
from requests import Session

from common.anker_api import api as api_module
from common.env_utils import load_repo_env_local

_LOGGER: logging.Logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables from env.local file if it exists."""
    load_repo_env_local(Path(__file__), overwrite=True)


@pytest.fixture
def api_client():
    """Fixture to provide an authenticated API client."""
    user = os.getenv("ANKERUSER")
    password = os.getenv("ANKERPASSWORD")
    country = os.getenv("ANKERCOUNTRY")
    if not (user and password and country):
        pytest.skip(
            "Set ANKERUSER, ANKERPASSWORD, and ANKERCOUNTRY (or env.local) to run API tests."
        )

    with Session() as websession:
        myapi = api_module.AnkerSolixApi(
            user,
            password,
            country,
            websession,
            _LOGGER,
        )
        myapi.authenticate()
        myapi.update_sites()
        myapi.update_site_details()
        myapi.update_device_details()
        myapi.update_device_energy()
        yield myapi


def test_authentication(api_client):
    """Test authentication."""
    myapi = api_client
    assert myapi.apisession._loggedIn is True  # pylint: disable=protected-access
    assert myapi.apisession._token is not None  # pylint: disable=protected-access


def test_energy_analysis_yesterday_files(api_client):
    """Fetch yesterday's energy analysis for key devTypes."""
    myapi = api_client
    system = list(myapi.sites.values())[0]
    siteid = system["site_info"]["site_id"]
    devicesn = system["solarbank_info"]["solarbank_list"][0]["device_sn"]
    yesterday = datetime.today() - timedelta(days=10)

    data = myapi.energy_analysis(
        siteId=siteid,
        deviceSn=devicesn,
        rangeType="day",
        startDay=yesterday,
        endDay=yesterday,
        devType="solar_production",
    )
    assert isinstance(data, dict)
