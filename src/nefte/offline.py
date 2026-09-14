"""Запрет сетевых соединений: доказательство, что система работает в закрытом контуре.

На сессии вопросов 11.09 организаторы назвали это критерием оценки: технологическая
сеть предприятия не связана с интернетом, и решение обязано работать локально.
Утверждение «у нас нет внешних вызовов» проверяется не чтением кода, а запуском:
внутри ``forbid_network()`` любая попытка разрешить имя или открыть соединение с
чем-либо, кроме локальной машины, падает с ``NetworkForbidden``. Локальные адреса
разрешены — дашборд и возможная внутренняя LLM работают на своей машине или в
своей сети, а не в интернете.

Пользуются этим тест ``tests/test_runs_offline.py`` и скрипт
``scripts/check_offline.py``.
"""
from __future__ import annotations

import contextlib
import ipaddress
import socket


class NetworkForbidden(RuntimeError):
    """Попытка выйти в сеть там, где сети нет."""


_LOCAL_NAMES = {"localhost", "localhost.localdomain", ""}


def _is_local(host) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    host = str(host).strip("[]")
    if host.lower() in _LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


@contextlib.contextmanager
def forbid_network(attempts: list | None = None):
    """Внутри блока соединения наружу запрещены; ``attempts`` собирает попытки."""
    original = (socket.socket.connect, socket.socket.connect_ex,
                socket.create_connection, socket.getaddrinfo)

    def _deny(what: str, host):
        if attempts is not None:
            attempts.append((what, host))
        raise NetworkForbidden(f"сетевой вызов в закрытом контуре: {what} {host!r}")

    def connect(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(host):
            _deny("connect", host)
        return original[0](self, address)

    def connect_ex(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(host):
            _deny("connect_ex", host)
        return original[1](self, address)

    def create_connection(address, *args, **kwargs):
        if not _is_local(address[0]):
            _deny("create_connection", address[0])
        return original[2](address, *args, **kwargs)

    def getaddrinfo(host, *args, **kwargs):
        if not _is_local(host):
            _deny("getaddrinfo", host)
        return original[3](host, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.create_connection = create_connection
    socket.getaddrinfo = getaddrinfo
    try:
        yield attempts
    finally:
        (socket.socket.connect, socket.socket.connect_ex,
         socket.create_connection, socket.getaddrinfo) = original
