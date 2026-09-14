"""Система работает без сети — проверено запуском, а не чтением кода.

Организаторы на сессии 11.09 назвали критерием оценки возможность запуска в
закрытой среде: технологическая сеть с интернетом не связана. Здесь цикл
принятия решения и импорт всех модулей системы идут внутри запрета на сетевые
соединения (``nefte.offline.forbid_network``). Отдельный тест проверяет сам
запрет: без него зелёный результат ничего бы не значил.

Прогон на настоящих данных и обученных моделях — ``scripts/check_offline.py``.
"""
from __future__ import annotations

import importlib
import pkgutil
import socket
from pathlib import Path

import pytest

import nefte
from nefte.offline import NetworkForbidden, forbid_network
from tests.test_agents import build_system, make_state

ROOT = Path(__file__).resolve().parents[1]


def test_the_guard_really_blocks_outside_connections():
    """Контрольный опыт: запрет обязан ловить и разрешение имени, и соединение."""
    attempts = []
    with forbid_network(attempts):
        with pytest.raises(NetworkForbidden):
            socket.getaddrinfo("pypi.org", 443)
        with pytest.raises(NetworkForbidden):
            socket.create_connection(("93.184.216.34", 80), timeout=0.1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(NetworkForbidden):
                sock.connect(("8.8.8.8", 53))
        finally:
            sock.close()
    assert len(attempts) == 3
    # после блока всё возвращается как было
    assert socket.getaddrinfo.__module__ == "socket"


def test_local_addresses_stay_allowed():
    """Дашборд и внутренняя LLM живут на своей машине — их запрет не трогает."""
    with forbid_network():
        socket.getaddrinfo("localhost", 0)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client.connect(server.getsockname())
        finally:
            client.close()
            server.close()


def test_every_module_imports_without_network():
    """Ни один модуль системы не ходит в сеть при импорте."""
    names = [m.name for m in pkgutil.walk_packages(nefte.__path__, "nefte.")]
    assert len(names) > 20, "обход пакета ничего не нашёл — проверка была бы пустой"
    with forbid_network():
        for name in names:
            importlib.import_module(name)


def test_decision_cycle_runs_without_network():
    """Весь цикл: качество, надёжность, оптимизатор, оркестратор, карточка."""
    attempts = []
    with forbid_network(attempts):
        system = build_system()
        for state in (make_state(lims=(5.0, 1.0), pak=(5.2, 0.1)),
                      make_state(lims=(11.0, 1.0), pak=(11.2, 0.1)),
                      make_state(usable=False)):
            rec = system.run(state)
            assert rec.to_operator_text()
    assert attempts == []


def test_dashboard_does_not_report_usage_to_streamlit():
    """Streamlit по умолчанию отправляет статистику использования наружу.

    В закрытой сети это не ломает работу, но исходящие попытки видит служба
    безопасности. Отключено в конфиге, который Streamlit читает из корня проекта.
    """
    config = ROOT / ".streamlit" / "config.toml"
    assert config.exists(), "нет .streamlit/config.toml"
    text = config.read_text(encoding="utf-8")
    assert "gatherUsageStats = false" in text
