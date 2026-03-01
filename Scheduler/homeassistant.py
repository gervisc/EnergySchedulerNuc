from typing import Any, Dict
import os
import logging
import requests

from common.anker_repository import AnkerRepository

LOGGER = logging.getLogger(__name__)

def _get_ha_config() -> tuple[str, str]:
    ha_url = os.getenv("HA_URL")
    ha_token = os.getenv("HA_TOKEN")
    if not ha_url or not ha_token:
        raise RuntimeError("Home Assistant config missing: set HA_URL and HA_TOKEN")
    return ha_url.rstrip("/"), ha_token

def ha_call_service(domain: str, service: str, payload: Dict[str, Any]) -> None:
    ha_url, ha_token = _get_ha_config()
    url = f"{ha_url}/api/services/{domain}/{service}"
    headers = {
        "Authorization": f"Bearer {ha_token}",
        "Content-Type": "application/json",
    }
    LOGGER.info("HA call: %s payload=%s", url, payload)
    r = requests.post(url, headers=headers, json=payload, timeout=15)
    if not r.ok:
        content_type = r.headers.get("Content-Type", "")
        response_body = r.text
        LOGGER.error(
            "HA error: %s status=%s reason=%s content_type=%s payload=%s response=%s",
            url,
            r.status_code,
            r.reason,
            content_type,
            payload,
            response_body,
        )
    r.raise_for_status()

def set_input_number(entity_id: str, value: float) -> None:
    ha_call_service("number", "set_value", {"entity_id": entity_id, "value": int(round(value))})

def set_input_select(entity_id: str, option: str) -> None:
    # option must be one of the configured options in HA for this input_select
    ha_call_service("select", "select_option", {"entity_id": entity_id, "option": option})

def update_battery_actions(
    setpoint: float,
    tarif_group: str,
    usage_mode: str,
) -> None:
    user = os.getenv("ANKERUSER")
    password = os.getenv("ANKERPASSWORD")
    country = os.getenv("ANKERCOUNTRY")
    site_id = os.getenv("SITE_ID")
    device_sn = os.getenv("DEVICE_SN")
    if not all([user, password, country, site_id, device_sn]):
        raise RuntimeError(
            "Missing Anker config: set ANKERUSER, ANKERPASSWORD, ANKERCOUNTRY, SITE_ID, DEVICE_SN"
        )

    repo = AnkerRepository(
        user=user,
        password=password,
        country=country,
        site_id=site_id,
        device_sn=device_sn,
        logger=LOGGER,
    )
    repo.update_battery_actions(
        setpoint=float(setpoint),
        tariff_group=str(tarif_group),
        usage_mode=str(usage_mode),
    )
