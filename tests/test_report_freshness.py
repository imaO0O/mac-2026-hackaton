"""Закоммиченные отчёты должны соответствовать текущему коду и конфигу.

README обещает, что числа воспроизводятся командой. Обещание легко потерять
молча: поменяли порог в конфиге или расчёт признаков — а отчёты в репозитории
остались прежними, и на защите звучат числа, которых код уже не даёт.

Такое в проекте случалось: `quality_metrics_h2.json` однажды не воспроизводился
командой из README, и выяснилось это случайно. Эти тесты ловят тот же класс
расхождений сразу.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nefte.config import ROOT, load_config
from nefte.models.dataset import FEATURE_VERSION

# Контракт покрывает ВСЕ отчёты, а не перечисленные по имени. Перечень по именам
# уже подводил: новый отчёт добавляли, в список его не вносили, и он оставался вне
# проверки — молча, потому что отсутствие файла в списке ничем не отличается от
# отсутствия расхождений.
REPORTS = sorted((ROOT / "reports").glob("*.json"))

# А эти обязаны нести версию матрицы, а не просто «нести, если несут». Разница
# принципиальная: отчёт без поля версии тесты ниже ПРОПУСКАЮТ, и пропуск выглядит
# как успех. Ровно так и вышло: докстринг контракта утверждал, что отчёты сетей
# покрыты, а они пропускались все девять — поле в scripts/train_sequence.py
# добавили в 18:59, когда прогон сетей уже шёл с 18:54 и работал по старому коду.
#
# Числа при этом были свежие, а контракт — неисполняемым. Список ниже превращает
# пропуск в падение: если семейство отчётов зависит от матрицы, оно обязано
# сказать, на какой матрице снято.
MUST_CARRY_VERSION = ("quality_metrics_", "sequence_metrics_", "alarm_budget_",
                      "test_period", "val_period", "simulation", "calibration_", "drift_",
                      "feature_stability_", "sensitivity_", "objective_weights",
                      "adversarial", "architectures", "severity_robustness",
                      "vak_vs_lims")

# Отчёты, где решения принимал оркестратор: тяжесть режима в них зависит от того,
# как агент надёжности мерил износ катализатора (configs → reliability). Без поля
# переключение выключателя оставило бы их «свежими» по всем проверкам выше, хотя
# числа в них посчитаны другой тяжестью режима.
MUST_CARRY_RELIABILITY = ("test_period.json", "test_period_step1h.json", "test_period_boost_",
                          "test_period_seq_", "simulation", "adversarial", "architectures",
                          "severity_robustness", "objective_weights", "return_to_base",
                          "kinetic_order", "reliability_metrics")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.name)
def test_report_matches_the_current_feature_version(path: Path):
    """Отчёт, снятый на другой матрице признаков, — не отчёт, а история.

    Версия поднимается, когда меняется САМ РАСЧЁТ признаков. Если она разошлась,
    числа в отчёте получены не тем кодом, который лежит рядом.
    """
    report = _load(path)
    version = report.get("feature_version")
    if version is None:
        pytest.skip("отчёт снят до того, как версия стала записываться")
    assert version == FEATURE_VERSION, (
        f"{path.name} снят на матрице версии {version}, а код даёт "
        f"{FEATURE_VERSION}: перезапустите scripts/train_quality.py")


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.name)
def test_report_matches_the_current_limits(path: Path):
    """Предел показателя и бюджет тревог берутся из конфига, а не из памяти."""
    cfg = load_config()
    report = _load(path)
    target = report.get("target", "sulfur")
    expected_limit = (cfg["spec"]["product_sulfur_mgkg"]["max"] if target == "sulfur"
                      else cfg["spec"]["t95_c"]["max"])
    if "limit" in report:
        assert report["limit"] == pytest.approx(float(expected_limit))
    if report.get("alarm_budget") is not None:
        assert report["alarm_budget"] == pytest.approx(
            float(cfg["quality"]["alarm_budget"]))


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.name)
def test_report_matches_the_current_split(path: Path):
    """Границы train/val/test — часть постановки, а не деталь прогона.

    Сдвинули разбиение — прежние метрики относятся к другому эксперименту.
    """
    cfg = load_config()
    split = _load(path).get("split")
    if split is None:
        pytest.skip("отчёт снят до того, как разбиение стало записываться")
    for name in ("train", "val", "test"):
        assert list(split[name]) == list(cfg["split"][name]), (
            f"{path.name}: разбиение {name} разошлось с конфигом")


@pytest.mark.parametrize(
    "path", [p for p in REPORTS if p.name.startswith(MUST_CARRY_VERSION)],
    ids=lambda p: p.name)
def test_report_that_depends_on_the_matrix_says_which_one(path: Path):
    """Отчёт, зависящий от матрицы, обязан назвать её версию.

    Без этого проверка свежести превращается в пропуск, а пропуск читается как
    успех. Это не гипотетическая опасность: девять отчётов сетей пропускались,
    пока докстринг рядом утверждал, что они покрыты.
    """
    assert _load(path).get("feature_version") is not None, (
        f"{path.name} не записывает версию матрицы — проверки свежести его "
        "молча пропускают, и устареть он может незаметно")


def test_there_are_reports_to_check():
    """Сам список не должен молча опустеть — иначе тесты выше ничего не проверяют."""
    assert REPORTS, "в reports/ нет ни одного отчёта"
    covered = [p for p in REPORTS if p.name.startswith(MUST_CARRY_VERSION)]
    assert covered, "ни один отчёт не попал под обязательную проверку версии"


def _reliability_settings(cfg: dict) -> dict:
    settings = cfg.get("reliability") or {}
    return {"catalyst_factor": settings.get("catalyst_factor", "age"),
            "catalyst_reset": settings.get("catalyst_reset", "outage_48h")}


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.name)
def test_report_matches_the_current_reliability_settings(path: Path):
    """Износ катализатора в severity мерится так, как сейчас записано в конфиге.

    Сравнения вариантов называют настройки в имени отчёта («catalyst_…») и
    намеренно с конфигом не совпадают.
    """
    settings = _load(path).get("reliability_settings")
    if settings is None:
        pytest.skip("отчёт не зависит от тяжести режима или снят до поля")
    if "catalyst_" in path.name:
        pytest.skip("сравнение вариантов с явными настройками")
    assert settings == _reliability_settings(load_config()), (
        f"{path.name} посчитан с износом катализатора {settings}, а конфиг даёт "
        f"{_reliability_settings(load_config())}: пересоберите отчёт")


# Не пересобраны после включения catalyst_factor: activity, потому что на машине
# пересборки нет моделей: сети в репозиторий не входят (models/ в .gitignore) и
# обучаются на GPU (docs/GPU_SETUP.md). Список явный и строгий: пересоберут отчёт —
# тест начнёт проходить, strict-xfail упадёт, и строку отсюда надо убрать.
KNOWN_STALE_RELIABILITY = {
    name: "прогон сети: модели нет на машине пересборки, пересобрать после train_sequence.py"
    for name in ("test_period_seq_h0.json", "test_period_seq_h2.json",
                 "test_period_seq_s100_h2.json", "test_period_seq_s200_h2.json")
}


def _orchestrator_reports() -> list:
    params = []
    for path in REPORTS:
        if not path.name.startswith(MUST_CARRY_RELIABILITY):
            continue
        reason = KNOWN_STALE_RELIABILITY.get(path.name)
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        params.append(pytest.param(path, marks=marks, id=path.name))
    return params


@pytest.mark.parametrize("path", _orchestrator_reports())
def test_report_built_by_the_orchestrator_says_how_wear_was_measured(path: Path):
    """Иначе проверка выше его пропускает, а пропуск читается как успех."""
    assert _load(path).get("reliability_settings") is not None, (
        f"{path.name} не записывает настройки износа катализатора")
