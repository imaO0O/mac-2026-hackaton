"""Каждая сцена защиты рендерится на дашборде без ошибки.

26.09 сцена «Недостоверные данные» падала с KeyError: карточка перестала обещать
диапазон следующего анализа на непригодном срезе, а дашборд ждал его всегда. Тесты
проверяли карточку и оркестратор, но не страницу — и падение дошло бы до демо.
Здесь страница рендерится так, как её увидит эксперт: `streamlit.testing.AppTest`
без браузера, по сцене за раз, плюс момент, на котором падение нашлось.
"""
from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from nefte.config import ROOT  # noqa: E402

APP = str(ROOT / "app" / "dashboard.py")


def _render(at):
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _scene_count() -> int:
    from streamlit.testing.v1 import AppTest

    at = _render(AppTest.from_file(APP, default_timeout=600))
    return len(at.sidebar.selectbox[0].options)


def test_every_defense_scene_renders():
    from streamlit.testing.v1 import AppTest

    for index in range(_scene_count()):
        at = AppTest.from_file(APP, default_timeout=600)
        at.run()
        at.sidebar.selectbox[0].set_value(index).run()
        assert not at.exception, (index, [e.value for e in at.exception])


def test_moment_with_unusable_data_renders():
    import datetime as dt

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    at.sidebar.radio[0].set_value("Любой момент").run()
    at.sidebar.date_input[0].set_value(dt.date(2026, 4, 26)).run()
    at.sidebar.slider[0].set_value(20).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("не обещаем" in str(m.value) for m in at.metric)
