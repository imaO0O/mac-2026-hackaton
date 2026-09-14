"""Цикл решения как локальный HTTP-сервис — без внешних зависимостей.

    python -m nefte.service                 # http://127.0.0.1:8080
    python -m nefte.service --port 9000 --host 0.0.0.0

Зачем. На производстве система — не скрипт, который запускают руками, а служба в
технологической сети: система сбора данных присылает срез состояния, служба
возвращает рекомендацию. Сеть закрытая, поэтому сервис написан на стандартной
библиотеке Python — доустанавливать ничего не нужно.

Граница интеграции — контракт ``ProcessState`` из ``agents/schemas.py``: адаптер к
системе сбора (исторической базе, OPC и т. п.) обязан собрать ровно такой срез.

Методы:

* ``GET /health`` — жив ли сервис и какая модель качества подключена;
* ``GET /decide?ts=2026-02-18T00:00`` — решение на момент истории (срез собирается
  из выданных файлов, как в демо);
* ``POST /decide`` — решение по присланному срезу ``ProcessState`` в JSON.

Ответ — ``Recommendation`` в JSON (с трассой по агентам), плюс ``outcome`` и
``text`` — карточка оператора.

Ограничение, которое надо назвать. Обученная модель качества берёт признаки из
матрицы по метке времени среза, то есть из истории. Для присланного среза с
моментом, которого в истории нет, модель прогноза не даст, и агент качества
перейдёт на персистенцию по измерениям из среза. Живой расчёт признаков по
потоку — отдельная задача (docs/PLAN.md, пункт 9).
"""
from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from nefte.agents.schemas import ProcessState, Recommendation

MAX_BODY_BYTES = 2_000_000


class DecisionService:
    """Держит собранную систему и отвечает на запросы решения.

    Оркестратор хранит состояние между циклами (время последнего воздействия),
    поэтому запросы к одному сервису обрабатываются по очереди.
    """

    def __init__(self, system=None, state_builder=None):
        self._system = system
        self._sb = state_builder
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        if self._system is not None:
            return
        from nefte.config import load_config
        from nefte.pipeline import StateBuilder
        from scripts.run_cycle import build_system

        cfg = load_config()
        self._sb = StateBuilder(cfg)
        self._system = build_system(self._sb, cfg)
        self._system.log_runs = False

    def health(self) -> dict:
        model = getattr(getattr(self._system, "quality", None), "model", None)
        source = getattr(model, "source_path", None)
        return {"status": "ok", "system_loaded": self._system is not None,
                "quality_model": None if source is None else str(source)}

    def decide_state(self, state: ProcessState) -> Recommendation:
        self._ensure()
        with self._lock:
            return self._system.run(state)

    def decide_at(self, ts: str) -> Recommendation:
        self._ensure()
        if self._sb is None:
            raise LookupError("срез по времени недоступен: сервис собран без данных")
        with self._lock:
            return self._system.run(self._sb.build(pd.Timestamp(ts)))


def _payload(rec: Recommendation) -> dict:
    return {"outcome": rec.outcome(), "text": rec.to_operator_text(),
            "recommendation": rec.model_dump(mode="json")}


def make_handler(service: DecisionService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "nefte-decision/1"

        def _send(self, status: HTTPStatus, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):      # тише в консоли и в тестах
            pass

        def do_GET(self):                        # noqa: N802
            url = urlparse(self.path)
            if url.path == "/health":
                return self._send(HTTPStatus.OK, service.health())
            if url.path == "/decide":
                ts = (parse_qs(url.query).get("ts") or [None])[0]
                if not ts:
                    return self._send(HTTPStatus.BAD_REQUEST,
                                      {"error": "нужен параметр ts, например ?ts=2026-02-18T00:00"})
                try:
                    return self._send(HTTPStatus.OK, _payload(service.decide_at(ts)))
                except (LookupError, ValueError) as err:
                    return self._send(HTTPStatus.BAD_REQUEST, {"error": str(err)})
            return self._send(HTTPStatus.NOT_FOUND, {"error": "нет такого метода"})

        def do_POST(self):                       # noqa: N802
            if urlparse(self.path).path != "/decide":
                return self._send(HTTPStatus.NOT_FOUND, {"error": "нет такого метода"})
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                return self._send(HTTPStatus.BAD_REQUEST,
                                  {"error": "тело запроса — срез ProcessState в JSON"})
            try:
                state = ProcessState.model_validate_json(self.rfile.read(length))
            except ValueError as err:
                return self._send(HTTPStatus.UNPROCESSABLE_ENTITY,
                                  {"error": "срез не соответствует контракту ProcessState",
                                   "detail": str(err)[:2000]})
            return self._send(HTTPStatus.OK, _payload(service.decide_state(state)))

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8080,
          service: DecisionService | None = None) -> ThreadingHTTPServer:
    """Создаёт сервер; запуск — ``server.serve_forever()``."""
    return ThreadingHTTPServer((host, port), make_handler(service or DecisionService()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Цикл решения как локальный HTTP-сервис")
    ap.add_argument("--host", default="127.0.0.1",
                    help="по умолчанию только локальная машина")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    service = DecisionService()
    print("Сборка системы: данные и модель…", flush=True)
    service._ensure()
    server = serve(args.host, args.port, service)
    print(f"Сервис решения: http://{args.host}:{args.port}  (GET /health, GET/POST /decide)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
