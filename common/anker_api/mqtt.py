"""Minimal upstream-style MQTT session for Anker Solix current-state reads.

This follows the upstream project design:
- broker/certificates are retrieved from the Anker API via get_mqtt_info()
- topics are constructed from mqtt_info + device metadata
- realtime data is requested via MQTT command publish

It intentionally implements only the pieces needed by this project to fetch
live SOC for Solarbank devices.
"""

from __future__ import annotations

from base64 import b64decode, b64encode
import contextlib
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import secrets
import ssl
import tempfile
import threading
from typing import Any


DEFAULT_MQTT_PORT = 8883
DEFAULT_REALTIME_TRIGGER_SECONDS = 300


SOLARBANK_SOC_FIELDS: dict[str, dict[str, dict[str, str]]] = {
    "A17C1": {
        "0405": {"ad": "battery_soc"},
        "0408": {"b0": "battery_soc"},
        "040a": {"a3": "main_battery_soc"},
    },
    "A17C2": {
        "0405": {"ad": "battery_soc"},
        "0408": {"b0": "battery_soc"},
        "040a": {"a3": "main_battery_soc"},
    },
    "A17C3": {
        "0405": {"ad": "battery_soc"},
        "0408": {"b0": "battery_soc"},
        "040a": {"a3": "main_battery_soc"},
    },
    "A17C5": {
        "0405": {"a6": "battery_soc"},
        "0408": {"a7": "battery_soc"},
        "040a": {"a3": "main_battery_soc"},
    },
}


class DeviceHexDataTypes:
    str = bytes.fromhex("00")
    ui = bytes.fromhex("01")
    sile = bytes.fromhex("02")
    var = bytes.fromhex("03")
    bin = bytes.fromhex("04")
    sfle = bytes.fromhex("05")


@dataclass
class ParsedMqttMessage:
    model: str
    device_sn: str
    msgtype: str
    values: dict[str, Any]


class AnkerSolixMqttSession:
    """Minimal MQTT session built from upstream Anker Solix MQTT flow."""

    def __init__(self, apisession: Any, logger: Any) -> None:
        self.apisession = apisession
        self.logger = logger
        self.client = None
        self.host: str | None = None
        self.port = DEFAULT_MQTT_PORT
        self.mqtt_info: dict[str, Any] = {}
        self._cert_files: list[str] = []
        self._ready = threading.Event()
        self._subscribed = threading.Event()
        self._latest_values: dict[str, dict[str, Any]] = {}

    def get_topic_prefix(self, device_dict: dict[str, Any], publish: bool = False) -> str:
        app_name = str(self.mqtt_info.get("app_name") or "")
        device_pn = str(
            device_dict.get("device_pn") or device_dict.get("product_code") or ""
        )
        device_sn = str(device_dict.get("device_sn") or "")
        if not app_name or not device_pn or not device_sn:
            return ""
        return f"{'cmd' if publish else 'dt'}/{app_name}/{device_pn}/{device_sn}/"

    def connect(self, timeout_seconds: float) -> None:
        try:
            import paho.mqtt.client as mqtt
            from paho.mqtt.enums import CallbackAPIVersion
        except ImportError as exc:
            raise RuntimeError("paho-mqtt is required for Anker MQTT support") from exc

        self.mqtt_info = self.apisession.get_mqtt_info() or {}
        self.host = self.mqtt_info.get("endpoint_addr") or None
        if not self.host:
            raise RuntimeError("Anker API did not return MQTT endpoint information")

        self.client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id=f"{self.mqtt_info.get('thing_name')}_{secrets.randbelow(99999):05}",
            clean_session=True,
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.client.on_subscribe = self._on_subscribe
        self.client.on_publish = self._on_publish

        self._create_cert_files()
        self.client.tls_set(
            ca_certs=self._cert_files[0],
            certfile=self._cert_files[1],
            keyfile=self._cert_files[2],
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLS_CLIENT,
        )
        self.client.connect(self.host, self.port, keepalive=max(30, int(timeout_seconds)))
        self.client.loop_start()

    def cleanup(self) -> None:
        if self.client is not None:
            with contextlib.suppress(Exception):
                self.client.disconnect()
            with contextlib.suppress(Exception):
                self.client.loop_stop()
        self.client = None
        for filename in self._cert_files:
            with contextlib.suppress(OSError):
                Path(filename).unlink()
        self._cert_files = []
        self._ready.clear()

    def request_device_soc(
        self,
        device_dict: dict[str, Any],
        timeout_seconds: float,
        realtime_trigger_seconds: int = DEFAULT_REALTIME_TRIGGER_SECONDS,
    ) -> float | None:
        self._ready.clear()
        self._subscribed.clear()
        self._latest_values.pop(str(device_dict.get("device_sn") or ""), None)
        self.connect(timeout_seconds=timeout_seconds)
        try:
            topic_prefix = self.get_topic_prefix(device_dict)
            if not topic_prefix:
                self.logger.debug(
                    "MQTT topic prefix could not be built for device=%s mqtt_info_keys=%s",
                    device_dict,
                    sorted(self.mqtt_info.keys()),
                )
                return None
            topic = f"{topic_prefix}#"
            result, mid = self.client.subscribe(topic)
            self.logger.debug(
                "MQTT subscribe requested topic=%s result=%s mid=%s",
                topic,
                result,
                mid,
            )
            self._subscribed.wait(timeout=max(2.0, min(timeout_seconds, 10.0)))
            self._publish_realtime_trigger(
                device_dict=device_dict,
                trigger_timeout_seconds=realtime_trigger_seconds,
            )
            self._ready.wait(timeout=timeout_seconds)
            values = self._latest_values.get(str(device_dict.get("device_sn") or "")) or {}
            battery_soc = values.get("battery_soc")
            if battery_soc is None:
                battery_soc = values.get("main_battery_soc")
            return float(battery_soc) if battery_soc is not None else None
        finally:
            self.cleanup()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self.logger.debug("Connected to Anker MQTT broker reason_code=%s", reason_code)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        self.logger.debug("Disconnected from Anker MQTT broker reason_code=%s", reason_code)

    def _on_subscribe(self, client, userdata, mid, reason_code_list, properties=None):
        self.logger.debug(
            "MQTT subscribed mid=%s reason_codes=%s",
            mid,
            [getattr(code, "value", code) for code in (reason_code_list or [])],
        )
        self._subscribed.set()

    def _on_publish(self, client, userdata, mid, reason_code, properties=None):
        self.logger.debug(
            "MQTT publish completed mid=%s reason_code=%s",
            mid,
            getattr(reason_code, "value", reason_code),
        )

    def _on_message(self, client, userdata, msg) -> None:
        self.logger.debug(
            "MQTT message received topic=%s payload_len=%s",
            str(msg.topic),
            len(msg.payload or b""),
        )
        try:
            parsed = self._parse_message(topic=str(msg.topic), payload=msg.payload)
        except Exception:
            self.logger.debug("Failed to parse MQTT message", exc_info=True)
            return
        if not parsed:
            return
        if "battery_soc" not in parsed.values and "main_battery_soc" not in parsed.values:
            return
        self._latest_values[parsed.device_sn] = parsed.values
        self._ready.set()

    def _publish_realtime_trigger(
        self,
        device_dict: dict[str, Any],
        trigger_timeout_seconds: int,
    ) -> None:
        hexbytes = self._build_realtime_trigger_command(trigger_timeout_seconds)
        payload = {
            "head": {
                "version": "1.0.0.1",
                "client_id": (
                    f"android-{self.mqtt_info.get('app_name')}-"
                    f"{self.mqtt_info.get('user_id')}-"
                    f"{self.mqtt_info.get('certificate_id')}"
                ),
                "sess_id": "1234-5678",
                "msg_seq": 1,
                "seed": 1,
                "timestamp": int(datetime.now().timestamp()),
                "cmd_status": 2,
                "cmd": 17,
                "sign_code": 1,
                "device_pn": str(device_dict.get("device_pn") or ""),
                "device_sn": str(device_dict.get("device_sn") or ""),
            },
            "payload": json.dumps(
                {
                    "device_sn": str(device_dict.get("device_sn") or ""),
                    "account_id": device_dict.get("owner_user_id")
                    or self.mqtt_info.get("user_id"),
                    "data": b64encode(hexbytes).decode("utf-8"),
                },
                separators=(",", ":"),
            ),
        }
        topic = f"{self.get_topic_prefix(device_dict, publish=True)}req"
        payload_str = json.dumps(payload, separators=(",", ":"))
        info = self.client.publish(topic=topic, payload=payload_str)
        self.logger.debug(
            "MQTT publish requested topic=%s mid=%s payload_len=%s",
            topic,
            getattr(info, "mid", None),
            len(payload_str),
        )

    def _create_cert_files(self) -> None:
        auth_cache_dir = Path(__file__).parent / "authcache"
        auth_cache_dir.mkdir(parents=True, exist_ok=True)
        self._cert_files = []
        for key in ("aws_root_ca1_pem", "certificate_pem", "private_key"):
            filename = str(auth_cache_dir / f"{self.apisession.email}_mqtt_{key}.crt")
            Path(filename).write_text(str(self.mqtt_info.get(key) or ""), encoding="utf-8")
            self._cert_files.append(filename)

    def _parse_message(self, topic: str, payload: bytes) -> ParsedMqttMessage | None:
        message = json.loads(payload.decode("utf-8"))
        outer_payload = json.loads(message.get("payload") or "{}")
        model = str(outer_payload.get("pn") or "")
        device_sn = str(outer_payload.get("sn") or "")
        topic_parts = topic.split("/")
        if not model and len(topic_parts) > 2:
            model = topic_parts[2]
        if not device_sn and len(topic_parts) > 3:
            device_sn = topic_parts[3]

        data_value = outer_payload.get("data") or outer_payload.get("trans")
        if not model or not device_sn or not isinstance(data_value, str):
            return None

        decoded = b64decode(data_value)
        if outer_payload.get("trans") is not None:
            return None

        return self._parse_hex_payload(model=model, device_sn=device_sn, data=decoded)

    def _parse_hex_payload(
        self,
        model: str,
        device_sn: str,
        data: bytes,
    ) -> ParsedMqttMessage | None:
        if model not in SOLARBANK_SOC_FIELDS or len(data) < 10:
            self.logger.debug(
                "MQTT parse skipped: unsupported model=%s or short payload len=%s raw=%s",
                model,
                len(data),
                data.hex(),
            )
            return None

        msgtype = data[7:9].hex()
        field_map = SOLARBANK_SOC_FIELDS[model].get(msgtype)
        if not field_map:
            self.logger.debug(
                "MQTT parse skipped: unsupported msgtype model=%s device_sn=%s msgtype=%s raw=%s",
                model,
                device_sn,
                msgtype,
                data.hex(),
            )
            return None

        first_field = data[9:10]
        idx = 9 if first_field in [bytes([value]) for value in range(0xA0, 0xAA)] else 10
        values: dict[str, Any] = {}
        parsed_fields: list[dict[str, Any]] = []
        while idx < len(data) - 1:
            field_name = data[idx : idx + 1].hex()
            if not field_name:
                break
            field_length = int.from_bytes(data[idx + 1 : idx + 2], byteorder="little")
            if field_length <= 0:
                break
            field_start = idx + 2
            field_end = field_start + field_length
            field_data = data[field_start:field_end]
            idx = field_end
            parsed_fields.append(
                {
                    "field_name": field_name,
                    "field_length": field_length,
                    "field_data_hex": field_data.hex(),
                    "mapped_name": field_map.get(field_name),
                }
            )

            if field_name not in field_map or not field_data:
                continue

            parsed_value = self._decode_field_value(field_data)
            if parsed_value is not None:
                values[field_map[field_name]] = parsed_value

        self.logger.debug(
            "MQTT parse result model=%s device_sn=%s msgtype=%s start_idx=%s mapped_fields=%s values=%s raw=%s",
            model,
            device_sn,
            msgtype,
            9 if first_field in [bytes([value]) for value in range(0xA0, 0xAA)] else 10,
            parsed_fields,
            values,
            data.hex(),
        )

        return ParsedMqttMessage(
            model=model,
            device_sn=device_sn,
            msgtype=msgtype,
            values=values,
        )

    def _decode_field_value(self, field_data: bytes) -> Any | None:
        field_type = field_data[:1]
        raw_value = field_data[1:]
        if field_type == DeviceHexDataTypes.ui:
            return int.from_bytes(raw_value, signed=False)
        if field_type == DeviceHexDataTypes.sile:
            return int.from_bytes(raw_value, byteorder="little", signed=True)
        if field_type == DeviceHexDataTypes.var:
            return int.from_bytes(raw_value, byteorder="little", signed=True)
        if field_type == DeviceHexDataTypes.str:
            return raw_value.decode("utf-8", errors="ignore").strip()
        return None

    def _build_realtime_trigger_command(self, timeout_seconds: int) -> bytes:
        payload = bytearray()
        payload.extend(bytes.fromhex("a10122"))
        payload.extend(bytes.fromhex("a2020101"))
        payload.extend(bytes.fromhex("a30503"))
        payload.extend(int(timeout_seconds).to_bytes(length=4, byteorder="little", signed=False))
        timestamp = int(datetime.now().timestamp())
        payload.extend(bytes.fromhex("fe0503"))
        payload.extend(timestamp.to_bytes(length=4, byteorder="little", signed=False))

        header = bytearray(bytes.fromhex("ff09"))
        total_length = len(payload) + 10 + 1
        header.extend(total_length.to_bytes(length=2, byteorder="little", signed=False))
        header.extend(bytes.fromhex("03000f0057"))

        message = header + payload
        checksum = 0
        for byte in message:
            checksum ^= byte
        message.append(checksum)
        return bytes(message)
