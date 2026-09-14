"""Необязательный слой локальной LLM: ответ оператору на «почему?» по решению.

Выключен по умолчанию (``configs/config.yaml → llm.enabled``). Система полностью
работает без него: рекомендацию, её числа и объяснение формируют агенты.

Зачем он всё-таки есть. Внутри технологической сети предприятия, по ответу
организаторов, работает локальная LLM с OpenAI-совместимым API (модели до ~30 млрд
параметров). Оператор задаёт вопросы своими словами — «почему расход, а не
температура?», «что будет, если не делать ничего?» — и шаблонная карточка на это
не отвечает. Модель может переформулировать решение под вопрос.

Правило, без которого слой был бы опасен: **модель не может принести ни одного
своего числа.** Ей передаётся только структура решения (проблема, действие,
эффект, ограничения, трасса агентов), и в ответе проверяется каждое число: оно
обязано встречаться в этой структуре (с точностью до формы записи — 0.41 и 41 %,
9.57 и 9,57). Если нашлось чужое число, ответ отклоняется, и оператор видит
шаблонное объяснение с пометкой, какое число не подтвердилось. Недоступность
сервиса — не ошибка системы, а тот же возврат к шаблону.

Подключение — стандартная библиотека, без клиента OpenAI: в закрытом контуре
доустанавливать нечего.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from nefte.agents.schemas import Recommendation

# цифры внутри идентификаторов («T11», «P13», «local_024») числами не считаются
NUMBER = re.compile(r"(?<![A-Za-zА-Яа-яЁё_\d])[-+−]?\d+(?:[.,]\d+)?")
ENUMERATION = re.compile(r"(?m)^\s*\d+[.)]\s")

SYSTEM_PROMPT = (
    "Ты помогаешь оператору установки гидроочистки понять рекомендацию системы. "
    "Отвечай по-русски, коротко и по делу. Используй ТОЛЬКО факты и числа из "
    "переданной структуры решения. Не добавляй своих чисел, оценок, диапазонов и "
    "предположений. Если в структуре нет ответа на вопрос, так и скажи."
)


@dataclass
class Answer:
    text: str
    from_llm: bool
    rejected_numbers: list[str] = field(default_factory=list)
    note: str = ""


def decision_context(rec: Recommendation) -> dict:
    """Всё, что модели разрешено знать о решении."""
    return {
        "время": str(rec.ts), "исход": rec.outcome(), "проблема": rec.problem,
        "действие": ({} if rec.action is None else
                     {t: d for t, d in rec.action.deltas.items() if abs(d) > 1e-6}),
        "новые значения": ({} if rec.action is None else
                           {t: v for t, v in rec.action.moves.items()
                            if abs(rec.action.deltas.get(t, 0.0)) > 1e-6}),
        "эффект": dict(rec.expected_effect),
        "проверенные ограничения": list(rec.checked_constraints),
        "уверенность": rec.confidence,
        "объяснение": rec.explanation,
        "отказ": rec.abstain_reason if rec.abstained else "",
        "путь решения": [{"агент": s.agent, "что сказал": s.summary} for s in rec.trace],
    }


def _numbers(text: str) -> list[str]:
    return NUMBER.findall(text)


def _as_float(token: str) -> float:
    return float(token.replace("−", "-").replace(",", "."))


def allowed_values(context: dict) -> list[float]:
    """Числа структуры решения — и в долях, и в процентах."""
    blob = json.dumps(context, ensure_ascii=False)
    values = [_as_float(t) for t in _numbers(blob)]
    return values + [v * 100 for v in values] + [v / 100 for v in values]


def ungrounded_numbers(answer: str, context: dict, question: str = "") -> list[str]:
    """Числа ответа, которых нет ни в решении, ни в самом вопросе оператора."""
    allowed = allowed_values(context) + [_as_float(t) for t in _numbers(question)]
    # нумерация пунктов в начале строки («1. », «2) ») чисел не несёт; любое другое
    # число, даже маленькое целое, проверяется — «3 мг/кг» тоже может быть выдумкой
    answer = ENUMERATION.sub("", answer)
    bad = []
    for token in _numbers(answer):
        value = _as_float(token)
        decimals = len(token.replace(",", ".").split(".")[1]) if re.search(r"[.,]", token) else 0
        tolerance = 0.5 * 10 ** -decimals + 1e-9
        if not any(abs(abs(value) - abs(a)) <= tolerance for a in allowed):
            bad.append(token)
    return bad


class LocalLLMExplainer:
    """Клиент OpenAI-совместимого ``/chat/completions`` с проверкой чисел."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 20.0,
                 api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.api_key = api_key

    @classmethod
    def from_config(cls, cfg: dict) -> "LocalLLMExplainer | None":
        llm = cfg.get("llm") or {}
        if not llm.get("enabled"):
            return None
        return cls(llm["base_url"], llm["model"], float(llm.get("timeout_s", 20.0)),
                   llm.get("api_key"))

    def _complete(self, messages: list[dict]) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0.0},
                          ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"]).strip()

    def ask(self, rec: Recommendation, question: str) -> Answer:
        fallback = rec.explanation or rec.abstain_reason or rec.problem
        context = decision_context(rec)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Структура решения:\n"
             + json.dumps(context, ensure_ascii=False, indent=1)
             + f"\n\nВопрос оператора: {question}"},
        ]
        try:
            text = self._complete(messages)
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as err:
            return Answer(fallback, from_llm=False,
                          note=f"локальная модель недоступна ({type(err).__name__}); "
                               "показано объяснение системы")
        bad = ungrounded_numbers(text, context, question)
        if bad:
            return Answer(fallback, from_llm=False, rejected_numbers=bad,
                          note="ответ модели отклонён: числа "
                               + ", ".join(bad) + " не найдены в решении; показано "
                               "объяснение системы")
        return Answer(text, from_llm=True)
