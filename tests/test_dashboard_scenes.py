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


def test_decision_banner_matches_outcome():
    """Плашка решения наверху говорит то же, что карточка: цвет — по исходу.

    Плашка — первое, что видит оператор, и её ошибка хуже ошибки в таблице ниже:
    «Режим не менять» зелёным на отказе по данным отправил бы его работать по
    ложным цифрам. Сцены защиты покрывают все пять исходов.
    """
    from streamlit.testing.v1 import AppTest

    expected = {"unit_down": "nk-down", "bad_data": "nk-stop", "stable": "nk-hold",
                "quality_risk": "nk-act"}
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from demo import SCENES

    keys = [s["key"] for s in SCENES for _ in (s["ts"] if isinstance(s["ts"], list)
                                             else [s["ts"]])]
    seen = set()
    for index, key in enumerate(keys):
        want = next((cls for prefix, cls in expected.items() if key.startswith(prefix)),
                    None)
        if want is None or want in seen:
            continue
        at = AppTest.from_file(APP, default_timeout=600)
        at.run()
        at.sidebar.selectbox[0].set_value(index).run()
        banners = [m.value for m in at.markdown if "class='nk-dec " in str(m.value)]
        assert banners and f"nk-dec {want}" in banners[0], (key, banners[:1])
        seen.add(want)
    assert seen == set(expected.values())
