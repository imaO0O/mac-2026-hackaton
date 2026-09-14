"""Слой локальной LLM: модель не может принести своё число.

Настоящей модели в тестах нет: на свободном порту локальной машины поднимается
заглушка OpenAI-совместимого ``/chat/completions`` с заранее заданным ответом.
Всё идёт внутри запрета на внешние соединения.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nefte.llm_explain import LocalLLMExplainer, decision_context, ungrounded_numbers
from nefte.offline import forbid_network
from tests.test_agents import build_system, make_state


def _fake_llm(reply: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):                        # noqa: N802
            length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(length))
            Handler.seen = request
            body = json.dumps({"choices": [{"message": {"content": reply}}]},
                              ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, Handler


@pytest.fixture()
def rec():
    return build_system().run(make_state(lims=(11.0, 1.0), pak=(11.2, 0.1)))


def _ask(rec, reply: str, question: str = "Почему так?"):
    server, handler = _fake_llm(reply)
    try:
        explainer = LocalLLMExplainer(f"http://127.0.0.1:{server.server_address[1]}/v1",
                                      "local-model", timeout_s=5)
        with forbid_network():
            return explainer.ask(rec, question), handler
    finally:
        server.shutdown()
        server.server_close()


def test_grounded_answer_is_passed_through(rec):
    risk = rec.problem.split("по сере: ")[1].split("%")[0] if "по сере: " in rec.problem else ""
    reply = f"Риск по сере {risk}%, поэтому система предлагает изменить режим."
    answer, handler = _ask(rec, reply)
    assert answer.from_llm, answer.note
    assert answer.text == reply
    # модели ушла только структура решения, а не сырые данные
    content = handler.seen["messages"][1]["content"]
    assert "Структура решения" in content and "telemetry" not in content


def test_invented_number_rejects_the_answer(rec):
    answer, _ = _ask(rec, "Сера через сутки опустится до 3.27 мг/кг.")
    assert not answer.from_llm
    assert "3.27" in answer.rejected_numbers
    assert answer.text == (rec.explanation or rec.abstain_reason or rec.problem)
    assert "отклонён" in answer.note


def test_unreachable_model_falls_back_to_the_system_explanation(rec):
    explainer = LocalLLMExplainer("http://127.0.0.1:9/v1", "local-model", timeout_s=1)
    with forbid_network():
        answer = explainer.ask(rec, "Почему?")
    assert not answer.from_llm and "недоступна" in answer.note


def test_number_forms_are_recognised():
    # явный контекст: в настоящем решении случайные числа сделали бы проверку
    # зависимой от того, что вышло у синтетической системы
    context = {"действие": {"T11": 0.4, "P13": 0.05}, "лучший": "local_024",
               "эффект": {"сера, мг/кг": 8.12, "выпуск, %": -2.37}, "уверенность": 0.93}
    assert ungrounded_numbers("сера 8,12 мг/кг, уверенность 93 %, выпуск −2.37 %", context) == []
    assert ungrounded_numbers("сера 8.5 мг/кг", context) == ["8.5"]
    # маленькое целое не проходит «как номер пункта», если это не нумерация
    assert ungrounded_numbers("сера опустится до 3 мг/кг", context) == ["3"]
    assert ungrounded_numbers("1. Сера 8.12 мг/кг.\n2. Выпуск -2.37 %.", context) == []


def test_disabled_by_default():
    from nefte.config import load_config

    assert LocalLLMExplainer.from_config(load_config()) is None
