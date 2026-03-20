import json
import logging
from base64 import b64encode

from common.anker_api.mqtt import AnkerSolixMqttSession


class _FakeApiSession:
    email = "user@example.com"

    def get_mqtt_info(self):
        return {
            "app_name": "anker_power",
            "user_id": "user-id",
            "certificate_id": "cert-id",
            "thing_name": "thing-name",
            "endpoint_addr": "aiot-mqtt-eu.anker.com",
            "aws_root_ca1_pem": "ca",
            "certificate_pem": "cert",
            "private_key": "key",
        }


def _build_message(hex_payload: bytes, topic: str) -> tuple[bytes, str]:
    pn = topic.split("/")[2]
    sn = topic.split("/")[3]
    payload = {
        "head": {"timestamp": 1},
        "payload": json.dumps(
            {
                "pn": pn,
                "sn": sn,
                "data": b64encode(hex_payload).decode("utf-8"),
            }
        ),
    }
    return json.dumps(payload).encode("utf-8"), topic


def test_parse_a17c2_0405_battery_soc():
    session = AnkerSolixMqttSession(_FakeApiSession(), logging.getLogger("test"))
    hex_payload = bytes.fromhex(
        "ff090f0003010f040500"
        "ad02014b"
        "00"
    )
    payload, topic = _build_message(hex_payload, "dt/anker_power/A17C2/SN123/param_info")

    parsed = session._parse_message(topic=topic, payload=payload)

    assert parsed is not None
    assert parsed.device_sn == "SN123"
    assert parsed.values["battery_soc"] == 75


def test_build_topic_prefix_uses_upstream_layout():
    session = AnkerSolixMqttSession(_FakeApiSession(), logging.getLogger("test"))
    session.mqtt_info = {"app_name": "anker_power"}

    topic = session.get_topic_prefix(
        {"device_pn": "A17C2", "device_sn": "SN123"},
        publish=False,
    )

    assert topic == "dt/anker_power/A17C2/SN123/"
