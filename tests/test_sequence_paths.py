# -*- coding: utf-8 -*-
"""Сидовый вариант сети не имеет права затирать основную модель.

Так и было: проверка устойчивости по сидам (``--seed-base 100``, ``200``) писала
модель в каталог основной конфигурации — имя ОТЧЁТА сид содержало, имя МОДЕЛИ нет.
В ``sulfur_seq_tcn48_pre_h2`` оказалась модель на сидах 200–202 при отчёте про
сиды 42–44, загрузчик выбрал её по валидации, и прогон по тестовому периоду
считался на сидовом варианте. Ни один тест этого не видел: каждая часть по
отдельности была правильной.
"""
import re

from nefte.models.sequence import DEFAULT_SEED_BASE, SulfurSequenceModel


def test_seed_variants_get_their_own_directory():
    main = SulfurSequenceModel.default_path(2, "tcn", 48, True)
    s100 = SulfurSequenceModel.default_path(2, "tcn", 48, True, seed_base=100)
    s200 = SulfurSequenceModel.default_path(2, "tcn", 48, True, seed_base=200)
    assert len({main, s100, s200}) == 3


def test_main_configuration_keeps_its_old_name():
    """Старые артефакты основной конфигурации не должны осиротеть."""
    path = SulfurSequenceModel.default_path(2, "tcn", 48, True,
                                            seed_base=DEFAULT_SEED_BASE)
    assert path.name == "sulfur_seq_tcn48_pre_h2"


def test_report_name_is_derived_from_model_name():
    """Отчёт и модель называются одинаково — train_sequence выводит одно из другого."""
    for seed, expected in ((DEFAULT_SEED_BASE, "tcn48_pre"), (100, "tcn48_pre_s100")):
        path = SulfurSequenceModel.default_path(2, "tcn", 48, True, seed_base=seed)
        tag = path.name.removeprefix("sulfur_seq_").rsplit("_h", 1)[0]
        assert tag == expected


def test_loader_ignores_seed_variants():
    """Выбирать среди сидов лучший по валидации значило бы подбирать сид."""
    pattern = r"_s\d+_h[\d.]+$"
    assert re.search(pattern, "sulfur_seq_tcn48_pre_s100_h2")
    assert re.search(pattern, "sulfur_seq_gru24_s200_h0.5")
    assert not re.search(pattern, "sulfur_seq_tcn48_pre_h2")
    assert not re.search(pattern, "sulfur_seq_tcn48_h2")
    import pathlib
    source = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_cycle.py"
    assert pattern in source.read_text(encoding="utf-8"), \
        "загрузчик в run_cycle.py больше не отсекает сидовые варианты этим правилом"
