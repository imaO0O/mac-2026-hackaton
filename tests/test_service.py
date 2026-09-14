"""Сервис решения: контракт по HTTP и работа без внешней сети.

Сервис поднимается на свободном порту локальной машины с синтетической системой —
выданные данные не нужны. Весь обмен идёт внутри запрета на внешние соединения:
локальные адреса разрешены, как и в технологической сети.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from nefte.offline import forbid_network
from nefte.service import DecisionService, serve
from tests.test_agents import build_system, make_state


@pytest.fixture()
def server():
    httpd = serve("127.0.0.1", 0, DecisionService(system=build_system()))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _post(url: str, body: bytes):
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))


def test_health(server):
    with forbid_network():
        status, body = _get(f"{server}/health")
    assert status == 200 and body["status"] == "ok" and body["system_loaded"]


def test_post_state_returns_recommendation_with_trace(server):
    state = make_state(lims=(11.0, 1.0), pak=(11.2, 0.1))
    with forbid_network():
        status, body = _post(f"{server}/decide", state.model_dump_json().encode("utf-8"))
    assert status == 200
    assert body["outcome"] in ("меняем уставки", "держим режим", "отказ")
    agents = [s["agent"] for s in body["recommendation"]["trace"]]
    assert agents[0] == "срез состояния" and agents[-1] == "оркестратор"
    assert body["text"].startswith("[2026-04-20 12:00]")


def test_bad_state_is_rejected_with_a_reason(server):
    status, body = _post(f"{server}/decide", b'{"ts": "2026-04-20T12:00:00"}')
    assert status == 422
    assert "ProcessState" in body["error"]


def test_decide_by_time_needs_data_and_says_so(server):
    status, body = _post(f"{server}/decide", b"")
    assert status == 400
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(f"{server}/decide?ts=2026-02-18T00:00")
    assert err.value.code == 400
