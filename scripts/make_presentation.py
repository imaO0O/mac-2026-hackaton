"""Презентация решения из отчётов: docs/presentation.pptx.

    pip install python-pptx==1.0.2          # только для этого скрипта, системе не нужен
    python scripts/make_presentation.py

Зачем скрипт, а не файл, собранный руками. Числа решения пересчитываются каждой
пересборкой, и презентация, набранная вручную, расходится с отчётами в первый же
день — README это уже проходил («числа в README пришлось править руками трижды»).
Здесь каждое число берётся из reports/*.json теми же функциями, которыми
tests/test_readme_matches_reports.py сверяет README, поэтому слайды, README и
отчёты не могут разойтись. Пересобрали отчёты — пересобрали презентацию.

Слайды — по плану защиты (docs/PLAN.md, «демонстрация»): что делает система, как
устроена, на каких данных, главные числа, по агенту на слайд, зачем четыре агента,
карточка, отказы, решения по правилам до счёта, ограничения, как проверить. В
заметках к каждому слайду — что говорить.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lxml import etree  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.chart.data import CategoryChartData  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION  # noqa: E402
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

import tests.test_readme_matches_reports as T  # noqa: E402
from nefte.config import ROOT  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

OUT = ROOT / "docs" / "presentation.pptx"

NAVY = RGBColor(0x1F, 0x2A, 0x44)
AMBER = RGBColor(0xD9, 0x82, 0x2B)
GREEN = RGBColor(0x2E, 0x7D, 0x4F)
RED = RGBColor(0xB0, 0x3A, 0x2E)
GRAY = RGBColor(0x6B, 0x72, 0x80)
LIGHT = RGBColor(0xF3, 0xF4, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x11, 0x18, 0x27)
FONT = "Segoe UI"
MONO = "Consolas"

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)


# --------------------------------------------------------------------------- #
# числа — из отчётов, теми же функциями, что сверяют README
# --------------------------------------------------------------------------- #

def report(name: str) -> dict | None:
    path = ROOT / "reports" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace("-", "−")


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}".replace("-", "−")


def plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def actions(n: int) -> str:
    return f"{n} {plural(n, 'вмешательство', 'вмешательства', 'вмешательств')}"


def numbers() -> dict:
    model, pak = T.split0("model"), T.split0("baseline_pak")
    arch = T.report("architectures.json")["конфигурации"]
    sim = T.simulation()
    paired = sim["сера_без_вмешательства_сим"]
    life = T.report("catalyst_life.json")
    rate = life["скорость_дезактивации"]
    rest = life["остаточный_ресурс_мес"]
    analogy = [row["оставалось, мес"] for row in rest.get("по аналогии", [])
               if not row.get("нижняя граница")]
    window = T.event_window()
    refusals = T.report("refusal_quality.json")["выборки"]["тест"]
    periods = T.report("data_periods.json")
    return {
        "mae": model["MAE"], "mae_pak": pak["MAE"], "auc": model["roc_auc"],
        "coverage": model["coverage_80"],
        # «модель точнее прибора» — снятый вывод (docs/DEFENSE_QUALITY.md): парная
        # разница на границе значимости, поэтому слайд говорит «лучше или не хуже»
        "pak_diff": T.report("headline_intervals.json")["числа"]["бустинг h0"]
        ["MAE минус MAE ПАК"],
        "auc_h2": T.report("quality_metrics_h2.json")["splits"]["test"]["model"]["roc_auc"],
        "react_over": 100 * window["перед_превышением"],
        "react_norm": 100 * window["перед_нормой"],
        "arch": arch,
        "sim_actions": sim["вмешательств"],
        "sim_over_without": paired["выше предела без нас, шагов"],
        "sim_over_with": paired["выше предела с нами, шагов"],
        "sim_t95": sim["Т95_наш_вклад"]["средний сдвиг"],
        "rate": rate["°C/мес"], "rate_ci": rate["95% ДИ"],
        "life_lo": min(analogy) if analogy else None, "life_hi": rest["по средней скорости"],
        "freeze_on_outage": T.report("reliability_metrics.json")["downtime"]["explained_%"],
        "conf_ratio": (T.confidence_ratio("валидация"), T.confidence_ratio("тест")),
        "kinetic": T.kinetic_variant("D")["обещанное снижение, мг/кг на градус"],
        "practice": T.report("kinetic_strength.json")["практика_заказчика_мг_кг_на_градус"],
        "refusals": refusals,
        "periods": periods,
    }


# Решения, принятые и отклонённые по правилам, записанным до счёта. Вердикт читается
# из отчёта проверки: пересчитали — слайд показывает новый.
def _flag(key: str):
    return lambda d: d.get(key)


DECISIONS = [
    ("Оперативный анализатор — Q21 вместо файлового ПАК", "analyzer_source.json",
     lambda d: d.get("выбор") == "q21"),
    ("Отклик серы на уставки — кинетика порядка 1.5", "kinetic_strength.json",
     lambda d: str(d.get("выбор", "")).startswith("D")),
    ("Запас по сере держится и при слабом отклике", "robust_recommendation.json",
     _flag("принята")),
    ("Детектор «полки» Q21 с допуском", "q21_shelf.json",
     lambda d: d.get("выбор, мг/кг") is not None),
    # (б) стоит в конфиге; с 22.09 правило пропускает и (в) — см. docs/CATALYST_FACTOR.md
    ("Износ катализатора в тяжести — возраст от замены по журналу", "catalyst_factor.json",
     lambda d: (d.get("прошли") or {}).get("б")),
    ("«Режим уже идёт — не складывайте воздействия»", "already_moving.json", _flag("принят")),
    ("Разбор причины прогноза по группам каналов", "risk_attribution.json", _flag("принят")),
    ("Тонны продукта до следующего анализа", "tonnes_at_risk.json", _flag("принят")),
    ("Самоконтроль по приходящим анализам", "self_audit.json", _flag("принят")),
    ("Расхождение двух анализаторов — строкой в карточке", "analyzer_disagreement.json",
     _flag("принят")),
    ("Отказ при расхождении анализаторов", "disagreement_decisions.json",
     lambda d: d.get("решение") != "только карточка"),
    ("Предупреждение по режиму АВТ за 2 часа", "avt_warning.json", _flag("принято")),
    ("Раннее предупреждение нейросетью за 2 часа", "early_warning.json", _flag("принят")),
    ("Ранжирование по цене ошибки вместо весов", "economic_ranking.json", _flag("принято")),
    ("Рост перепада Р-202 в тяжести режима", "dp_growth.json", _flag("принят")),
    ("Возврат к базовому режиму после вмешательства", "return_to_base.json", _flag("принят")),
    ("План из нескольких шагов в карточке", "action_plan.json", _flag("принят")),
    ("Q21 как отдельный источник риска", "q21_source.json", _flag("принят")),
    ("Т95 комбинацией оценок вместо прошлого анализа", "t95_combo.json",
     lambda d: not str(d.get("выбор", "")).startswith("прошлый")),
    ("Т95 по формуле ВАК или модели", "t95_source.json",
     lambda d: d.get("выбор") != "прошлый анализ"),
    ("Высокая тяжесть вычёркивает только то, что греет", "severity_veto.json",
     _flag("принят")),
    ("Т95 по верхней границе доверия в ограничении", "t95_conservative.json",
     _flag("принят")),
    ("Границы уставок с поправкой на старение катализатора", "aging_bounds.json",
     _flag("принято")),
]


def verdicts() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    accepted, refused = [], []
    for label, name, verdict in DECISIONS:
        data = report(name)
        if data is None:
            continue
        (accepted if verdict(data) else refused).append((label, f"reports/{name}"))
    return accepted, refused


def card_example() -> tuple[str, str]:
    """Карточка из README — та же, что читает эксперт; README сверяется тестами."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    cmd = re.search(r"Пример карточки \(`([^`]+)`", text)
    block = re.search(r"Пример карточки.*?```\n(.*?)```", text, re.S)
    return (cmd.group(1) if cmd else ""), (block.group(1).rstrip() if block else "")


# --------------------------------------------------------------------------- #
# вёрстка
# --------------------------------------------------------------------------- #

class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.n = 0
        self.stamp = dt.date.today().strftime("%d.%m.%Y")

    def slide(self, title: str | None, notes: str = ""):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.n += 1
        if title:
            self.text(s, MARGIN, Inches(0.35), W - 2 * MARGIN, Inches(0.8),
                      [(title, 26, True, NAVY)])
            bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, Inches(1.12),
                                     Inches(1.2), Inches(0.06))
            self.fill(bar, AMBER)
            self.text(s, MARGIN, H - Inches(0.45), W - 2 * MARGIN, Inches(0.3),
                      [(f"МАС «Нефтекод» · числа из reports/, собрано {self.stamp}",
                        10, False, GRAY)])
            self.text(s, W - MARGIN - Inches(0.6), H - Inches(0.45), Inches(0.6),
                      Inches(0.3), [(str(self.n), 10, False, GRAY)], align=PP_ALIGN.RIGHT)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    @staticmethod
    def fill(shape, color, line=None):
        shape.fill.solid()
        shape.fill.fore_color.rgb = color
        if line is None:
            shape.line.fill.background()
        else:
            shape.line.color.rgb = line
        shape.shadow.inherit = False

    @staticmethod
    def text(s, x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             font=FONT, spacing=None):
        box = s.shapes.add_textbox(x, y, w, h)
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Inches(0.05)
        for i, para in enumerate(paras):
            text, size, bold, color = para[:4]
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            if spacing:
                p.space_after = Pt(spacing)
            run = p.add_run()
            run.text = text
            run.font.size, run.font.bold = Pt(size), bold
            run.font.color.rgb = color
            run.font.name = para[4] if len(para) > 4 else font
        return box

    def bullets(self, s, x, y, w, h, items, size=16, color=INK, gap=8):
        paras = [("•  " + item, size, False, color) for item in items]
        return self.text(s, x, y, w, h, paras, spacing=gap)

    def tile(self, s, x, y, w, h, big, small, color=NAVY):
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
        box.adjustments[0] = 0.08
        self.fill(box, LIGHT)
        self.text(s, x + Inches(0.15), y + Inches(0.12), w - Inches(0.3), Inches(0.8),
                  [(big, 30, True, color)])
        self.text(s, x + Inches(0.15), y + Inches(0.95), w - Inches(0.3), h - Inches(1.0),
                  [(small, 13, False, INK)])

    def node(self, s, x, y, w, h, title, sub, color=NAVY):
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
        box.adjustments[0] = 0.12
        self.fill(box, color)
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for i, (text, size, bold) in enumerate([(title, 15, True), (sub, 11, False)]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            run = p.add_run()
            run.text = text
            run.font.size, run.font.bold, run.font.name = Pt(size), bold, FONT
            run.font.color.rgb = WHITE
        return box

    @staticmethod
    def arrow(s, x1, y1, x2, y2, color=GRAY):
        line = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
        line.line.color.rgb = color
        line.line.width = Pt(1.75)
        ln = line.line._get_or_add_ln()
        tail = etree.SubElement(ln, qn("a:tailEnd"))
        tail.set("type", "triangle")
        tail.set("w", "med")
        tail.set("len", "med")
        return line

    def table(self, s, x, y, w, rows, widths, size=12, header=NAVY, row_h=0.36,
              colors=None):
        shape = s.shapes.add_table(len(rows), len(rows[0]), x, y, w,
                                   Inches(row_h * len(rows)))
        table = shape.table
        for i, width in enumerate(widths):
            table.columns[i].width = Inches(width)
        for r, row in enumerate(rows):
            table.rows[r].height = Inches(row_h)
            for c, value in enumerate(row):
                cell = table.cell(r, c)
                cell.margin_left = cell.margin_right = Inches(0.08)
                cell.margin_top = cell.margin_bottom = Inches(0.03)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.text = str(value)
                run = cell.text_frame.paragraphs[0].runs[0]
                run.font.size, run.font.name = Pt(size), FONT
                cell.fill.solid()
                if r == 0:
                    cell.fill.fore_color.rgb = header
                    run.font.bold, run.font.color.rgb = True, WHITE
                else:
                    cell.fill.fore_color.rgb = LIGHT if r % 2 else WHITE
                    run.font.color.rgb = (colors or {}).get((r, c), INK)
        return table

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(path)


# --------------------------------------------------------------------------- #
# слайды
# --------------------------------------------------------------------------- #

def title_slide(d: Deck):
    s = d.slide(None, notes=(
        "Одна фраза: система подсказывает оператору гидроочистки, как удержать серу и "
        "Т95 в спецификации, и честно молчит, когда данным нельзя верить."))
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    d.fill(band, NAVY)
    accent = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, Inches(3.55), Inches(1.6),
                                Inches(0.08))
    d.fill(accent, AMBER)
    d.text(s, MARGIN, Inches(1.7), W - 2 * MARGIN, Inches(1.2),
           [("МАС «Нефтекод»", 48, True, WHITE)])
    d.text(s, MARGIN, Inches(2.75), W - 2 * MARGIN, Inches(0.8),
           [("Мультиагентный советчик оператора гидроочистки дизельного топлива", 22,
             False, WHITE)])
    d.text(s, MARGIN, Inches(3.85), W - 2 * MARGIN, Inches(1.6),
           [("АВТ → гидроочистка 24-2000 → смешение", 18, False, LIGHT),
            ("Качество и технологические ограничения — выше экономики. "
             "Отказ с причиной — тоже решение.", 18, False, LIGHT)], spacing=6)
    d.text(s, MARGIN, H - Inches(0.9), W - 2 * MARGIN, Inches(0.4),
           [(f"Отборочный этап · числа из отчётов на {d.stamp} · работает локально, "
             "без интернета", 12, False, LIGHT)])


def what_slide(d: Deck):
    s = d.slide("Что делает система", notes=(
        "Три действия — оценить, сравнить, посоветовать или отказаться. Подчеркнуть: "
        "вариант, нарушающий жёсткое ограничение, отсекается ДО ранжирования и не может "
        "быть «оплачен» выпуском."))
    cols = [("1. Оценивает", "риск выхода продукта за спецификацию по сере и Т95 — "
             "с интервалом и уверенностью, по возрасту каждого анализа"),
            ("2. Сравнивает", "допустимые изменения режима: жёсткие ограничения — до "
             "ранжирования, затем фронт Парето по качеству, выпуску, энергии, тяжести"),
            ("3. Советует или молчит", "карточка оператору: причина → действие → эффект "
             "→ ограничения → уверенность → что дальше; или отказ с названной причиной")]
    x, w = MARGIN, (W - 2 * MARGIN - Inches(0.6)) / 3
    for head, body in cols:
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.6), w, Inches(2.6))
        box.adjustments[0] = 0.06
        d.fill(box, LIGHT)
        d.text(s, x + Inches(0.2), Inches(1.75), w - Inches(0.4), Inches(0.6),
               [(head, 20, True, NAVY)])
        d.text(s, x + Inches(0.2), Inches(2.4), w - Inches(0.4), Inches(1.8),
               [(body, 15, False, INK)])
        x += w + Inches(0.3)
    d.bullets(s, MARGIN, Inches(4.55), W - 2 * MARGIN, Inches(2.4), [
        "Работает в закрытом контуре: без интернета и внешних сервисов, на CPU; слой "
        "локальной LLM необязателен и выключен",
        "Каждая включённая настройка включена правилом, записанным до счёта; "
        "отклонённые записаны с числами так же подробно",
        "Числа документов сверяются с отчётами тестами: переписанная фраза или "
        "пересчитанный отчёт роняют проверку",
    ], size=16)


def architecture_slide(d: Deck):
    s = d.slide("Как устроено: агенты и обмен между ними", notes=(
        "Агенты обмениваются типизированными структурами (schemas.py), каждый цикл "
        "целиком пишется в журнал. Оркестратор проходит проверки в порядке важности: "
        "стоит ли установка, можно ли верить данным, есть ли допустимые варианты, "
        "нужен ли шаг вообще."))
    bw, bh = Inches(2.35), Inches(1.0)
    y_mid = Inches(3.2)
    src = d.node(s, MARGIN, y_mid, bw, bh, "Срез состояния",
                 "телеметрия, ЛИМС, ПАК/Q21; возраст анализов, брак", GRAY)
    q = d.node(s, Inches(3.45), Inches(1.75), bw, bh, "Агент качества",
               "сера и Т95: прогноз, интервал, риск, уверенность")
    r = d.node(s, Inches(3.45), Inches(4.65), bw, bh, "Агент надёжности",
               "тяжесть режима, останов, износ катализатора")
    o = d.node(s, Inches(6.3), y_mid, bw, bh, "Оптимизатор",
               "варианты → отсев по ограничениям → Парето")
    b = d.node(s, Inches(6.3), Inches(5.35), bw, Inches(0.85), "Смешение",
               "рецептура, доли = 100 %")
    x = d.node(s, Inches(9.15), y_mid, Inches(1.7), bh, "Оркестратор", "порядок проверок",
               AMBER)
    c = d.node(s, Inches(11.1), Inches(2.45), Inches(1.65), Inches(1.0), "Карточка",
               "или отказ с причиной", GREEN)
    j = d.node(s, Inches(11.1), Inches(4.0), Inches(1.65), Inches(1.0), "Журнал",
               "reports/runs", GRAY)

    def right(n):
        return n.left + n.width, n.top + n.height // 2

    def left(n):
        return n.left, n.top + n.height // 2

    for a, z in [(src, q), (src, r), (q, o), (r, o), (o, x)]:
        d.arrow(s, *right(a), *left(z))
    d.arrow(s, o.left + o.width // 2, o.top + o.height, b.left + b.width // 2, b.top)
    d.arrow(s, *right(b), x.left + x.width // 2, x.top + x.height)
    d.arrow(s, q.left + q.width, q.top + Inches(0.2), x.left + x.width // 2, x.top)
    d.arrow(s, *right(x), *left(c))
    d.arrow(s, *right(x), *left(j))
    d.text(s, MARGIN, Inches(1.45), Inches(2.7), Inches(1.4),
           [("Контракты агентов — типизированные структуры:", 12, False, GRAY),
            ("src/nefte/agents/schemas.py", 12, False, GRAY, MONO)])


def data_slide(d: Deck, n: dict):
    s = d.slide("Данные: как разбиты и где они неисправны", notes=(
        "Разбиение только по времени. Неисправные периоды НЕ вырезаны: система в них "
        "отказывается, называя причину, и все метрики посчитаны вместе с ними. "
        "Полный список с датами — docs/DATA_PERIODS.md."))
    split = n["periods"]["split"]
    d.table(s, MARGIN, Inches(1.5), Inches(5.6), [
        ["Выборка", "Период", "Для чего"],
        ["обучение", f"{split['train'][0]} … {split['train'][1]}", "модели"],
        ["валидация", f"{split['val'][0]} … {split['val'][1]}", "пороги и правила приёмки"],
        ["тест", f"{split['test'][0]} … {split['test'][1]}", "только отчёт"],
    ], [1.3, 2.5, 1.8], size=13)
    hours: dict[str, dict[str, float]] = {}
    for row in n["periods"]["часов по выборкам"]:
        hours.setdefault(row["что"], {})[row["выборка"]] = row["часов"]
    kinds = sorted(hours)
    splits = ["обучение", "валидация", "тест"]
    rows = [["Неисправность, часов", *splits]]
    for kind in kinds:
        rows.append([kind, *[f"{hours[kind].get(sp, 0):.0f}" for sp in splits]])
    d.table(s, Inches(6.5), Inches(1.5), Inches(6.2), rows, [3.0, 1.15, 1.2, 0.85], size=12)
    d.bullets(s, MARGIN, Inches(4.0), W - 2 * MARGIN, Inches(3.0), [
        "Лабораторный анализ публикуется с задержкой ~4 ч: до публикации система его "
        "не видит, у каждого анализа — возраст",
        "Поточные анализаторы уходят на «полку»: файловый ПАК стоит ровно на 18.45, Q21 — "
        "около 24.88 с дрожанием ±0.04; детектор залипания с допуском найден 21.09",
        f"{n['freeze_on_outage']:.0f} % «зависаний» поточного анализатора приходятся на "
        "остановы — останов распознаётся отдельно",
        "Данные организаторов не изменены; периоды с датами и причинами — "
        "docs/DATA_PERIODS.md",
    ], size=15)


def numbers_slide(d: Deck, n: dict):
    s = d.slide("Главные числа на тесте — 2026 год, в подборе не участвовал", notes=(
        "Каждое число — из отчёта в reports/, README сверяется с ними тестом. "
        "Тест — 2026-01-01 … 2026-08-07."))
    full, single = n["arch"]["полная"], n["arch"]["одноагентная"]
    tiles = [
        (f"{fmt(n['mae'])} мг/кг", f"MAE прогноза серы — не хуже поточного анализатора "
         f"({fmt(n['mae_pak'])}, разница на границе значимости); ROC-AUC риска {fmt(n['auc'])}"),
        (f"{n['react_over']:.0f} % / {n['react_norm']:.0f} %",
         "реакция системы до публикации анализа: пробы с превышением против нормальных"),
        (f"в {single['суммарно °C'] / full['суммарно °C']:.0f} раз",
         f"меньше двигает уставки, чем одноагентная система: {full['суммарно °C']:.1f} °C "
         f"против {single['суммарно °C']:.0f} за полгода"),
        (f"{n['sim_over_without'] - n['sim_over_with']} из {n['sim_over_without']}",
         f"шагов с превышением убирают {actions(n['sim_actions'])} за месяц в "
         f"замкнутой имитации; вклад в Т95 {signed(n['sim_t95'])} °C"),
        (f"{n['rate']:.2f} °C/мес", f"теряет активность катализатор (95 % ДИ "
         f"{n['rate_ci'][0]:.2f}–{n['rate_ci'][1]:.2f}); ресурс цикла "
         f"{n['life_lo']:.0f}–{n['life_hi']:.0f} мес"),
        (f"в {n['conf_ratio'][0]:.1f}–{n['conf_ratio'][1]:.1f} раза",
         "больше ошибка прогноза, когда карточка пишет уверенность ниже 0.7"),
    ]
    tw, th = (W - 2 * MARGIN - Inches(0.6)) / 3, Inches(2.4)
    for i, (big, small) in enumerate(tiles):
        x = MARGIN + (tw + Inches(0.3)) * (i % 3)
        y = Inches(1.5) + (th + Inches(0.3)) * (i // 3)
        d.tile(s, x, y, tw, th, big, small)


def quality_slide(d: Deck, n: dict):
    s = d.slide("Агент качества — виртуальный анализатор", notes=(
        "Главная мысль: от показания прибора модель зависит почти один к одному, от "
        "температуры реактора — почти никак, потому что контуры регулирования гасят "
        "отклик в истории. Поэтому приращение от уставок считает кинетика, и карточка "
        "это говорит. Прогноз на 2 часа не работает — сказать самим."))
    d.bullets(s, MARGIN, Inches(1.5), Inches(7.4), Inches(5.4), [
        f"MAE {fmt(n['mae'])} мг/кг против {fmt(n['mae_pak'])} у поточного анализатора — "
        f"лучше или не хуже: парная разница {fmt(n['pak_diff']['значение'])}, 90 % интервал "
        f"{fmt(n['pak_diff']['90% интервал'][0])}…{fmt(n['pak_diff']['90% интервал'][1])}; "
        f"ROC-AUC риска {fmt(n['auc'])}; 80-процентный интервал накрывает "
        f"{100 * n['coverage']:.0f} % анализов",
        f"В окне от 2 ч до отбора до публикации анализа система реагирует на "
        f"{n['react_over']:.0f} % проб с превышением против {n['react_norm']:.0f} % нормальных",
        "Отклик серы на уставки из истории не выучить — его гасят регуляторы. Приращение "
        f"считает кинетика: порядок 1.5, {fmt(n['kinetic'])} мг/кг на градус — внутри "
        f"{fmt(n['practice'][0], 1)}–{fmt(n['practice'][1], 1)}, названных технологом",
        "Т95 — уровень из последнего анализа; ограничение «не ухудшать», а не гарантия",
        f"Прогноз на 2 часа вперёд у бустинга не работает (ROC-AUC {fmt(n['auc_h2'])}) — "
        "и мы это пишем, а не прячем",
    ], size=16, gap=10)
    tw = Inches(4.3)
    d.tile(s, W - MARGIN - tw, Inches(1.5), tw, Inches(2.3), f"{fmt(n['mae'])} мг/кг",
           f"MAE на тесте; у поточного анализатора {fmt(n['mae_pak'])} — лучше или не хуже")
    d.tile(s, W - MARGIN - tw, Inches(4.1), tw, Inches(2.3),
           f"{fmt(n['kinetic'])} мг/кг/°C",
           "отклик по кинетике порядка 1.5 — середина практики технолога")
    d.text(s, MARGIN, Inches(6.55), W - 2 * MARGIN, Inches(0.4),
           [("docs/QUALITY_AGENT.md · reports/quality_metrics_h0.json · "
             "docs/SULFUR_RESPONSE.md", 12, False, GRAY)])


def reliability_slide(d: Deck, n: dict):
    s = d.slide("Агент надёжности: тяжесть, останов, катализатор", notes=(
        "Единственное место, где система говорит не про текущий момент, а про месяцы "
        "вперёд, и единственная её оценка, проверенная бэктестом на цикле с известным "
        "исходом. Замена катализатора 23.04.2026 — из журнала."))
    d.bullets(s, MARGIN, Inches(1.5), Inches(7.4), Inches(5.4), [
        f"Катализатор теряет {n['rate']:.2f} °C/мес активности (95 % ДИ "
        f"{n['rate_ci'][0]:.2f}–{n['rate_ci'][1]:.2f}) — по двум завершённым циклам",
        f"Ресурс текущего цикла — {n['life_lo']:.0f}–{n['life_hi']:.0f} месяцев; метод "
        "проверен бэктестом на цикле с известным исходом",
        "Тяжесть режима — по прокси (температура, давление, перепад), с износом "
        "катализатора по возрасту от замены; ограничивает уставки: при высокой тяжести "
        "повышать температуру нельзя",
        f"Останов распознаётся отдельно: {n['freeze_on_outage']:.0f} % «зависаний» "
        "анализатора — это остановы, а не поломки прибора",
        "Полка Q21 на 24.88 мг/кг — неисправность прибора: детектор с допуском, правило "
        "приёмки записано до счёта",
    ], size=16, gap=10)
    tw = Inches(4.3)
    d.tile(s, W - MARGIN - tw, Inches(1.5), tw, Inches(2.3), f"{n['rate']:.2f} °C/мес",
           "скорость дезактивации катализатора")
    d.tile(s, W - MARGIN - tw, Inches(4.1), tw, Inches(2.3),
           f"{n['life_lo']:.0f}–{n['life_hi']:.0f} мес", "остаточный ресурс цикла")
    d.text(s, MARGIN, Inches(6.55), W - 2 * MARGIN, Inches(0.4),
           [("docs/CATALYST_LIFE.md · docs/RELIABILITY_AGENT.md · reports/catalyst_life.json",
             12, False, GRAY)])


def optimizer_slide(d: Deck, n: dict):
    s = d.slide("Оптимизатор: сначала ограничения, потом выбор", notes=(
        "Имитация замкнутого контура: что будет, если оператор послушается. Шаги с "
        "превышением считаются по одной и той же имитации с нами и без нас."))
    d.bullets(s, MARGIN, Inches(1.5), Inches(7.4), Inches(5.4), [
        "Жёсткие ограничения — до ранжирования: сера с запасом на неопределённость, Т95 "
        "не хуже, уставки в рабочем диапазоне, шаг за цикл ограничен",
        "Запас по сере обязан держаться и при втрое более слабом отклике — робастная "
        "гарантия, включена по правилу до счёта",
        "Фронт Парето по качеству, выпуску, энергии и тяжести; запрет частых "
        "воздействий: скорость изменения режима ≤ 0.5 °C/ч",
        "Ни одного допустимого варианта — отказ с перечнем причин, а не «лучший из "
        "плохих»",
        f"Замкнутая имитация июня 2026: {actions(n['sim_actions'])} за месяц "
        f"убирают {n['sim_over_without'] - n['sim_over_with']} из "
        f"{n['sim_over_without']} шагов с превышением, вклад в Т95 {signed(n['sim_t95'])} °C",
    ], size=16, gap=10)
    tw = Inches(4.3)
    d.tile(s, W - MARGIN - tw, Inches(1.5), tw, Inches(2.3),
           f"{n['sim_over_without'] - n['sim_over_with']} из {n['sim_over_without']}",
           "шагов с превышением убраны в имитации", GREEN)
    d.tile(s, W - MARGIN - tw, Inches(4.1), tw, Inches(2.3), f"{signed(n['sim_t95'])} °C",
           "наш вклад в Т95 — второй обязательный показатель не ухудшен")
    d.text(s, MARGIN, Inches(6.55), W - 2 * MARGIN, Inches(0.4),
           [("docs/OPTIMIZER_AGENT.md · docs/HARD_CHECKS.md §6 · reports/simulation.json",
             12, False, GRAY)])


def architectures_slide(d: Deck, n: dict):
    s = d.slide("Зачем четыре агента: урезанные системы на том же тесте", notes=(
        "Читать парами: пропуски — ложные тревоги — суммарный ход уставок. Конфигурация, "
        "которая «ловит всё», обычно просто дёргает уставки постоянно. Одноагентная — "
        "только прогноз и порог."))
    names = ["полная", "без надёжности", "без оптимизатора", "одноагентная"]
    rows = [["Конфигурация", "Вмешательств", "Отказов", "Пропуски", "Ложные",
             "Ход, °C"]]
    for name in names:
        a = n["arch"][name]
        rows.append([name, a["вмешательств"], a["отказов"], f"{a['доля пропусков']:.2f}",
                     f"{a['доля ложных']:.2f}", f"{a['суммарно °C']:.1f}"])
    d.table(s, MARGIN, Inches(1.5), Inches(7.2), rows, [1.8, 1.3, 0.9, 1.05, 0.95, 1.2],
            size=11, row_h=0.42)
    chart_data = CategoryChartData()
    chart_data.categories = list(reversed(names))
    chart_data.add_series("Ход уставок, °C",
                          [n["arch"][k]["суммарно °C"] for k in reversed(names)])
    frame = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(8.1), Inches(1.4),
                               Inches(4.6), Inches(3.4), chart_data)
    chart = frame.chart
    chart.has_legend = False
    chart.has_title = True
    chart.chart_title.text_frame.text = "Суммарный ход уставок, °C"
    chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
    plot = chart.plots[0]
    plot.gap_width = 60
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.number_format, labels.number_format_is_linked = "0", False
    labels.position = XL_LABEL_POSITION.OUTSIDE_END
    labels.font.size = Pt(11)
    series = plot.series[0]
    for i, name in enumerate(reversed(names)):
        point = series.points[i]
        point.format.fill.solid()
        point.format.fill.fore_color.rgb = AMBER if name == "полная" else GRAY
    chart.value_axis.visible = False
    chart.value_axis.has_major_gridlines = False
    chart.category_axis.tick_labels.font.size = Pt(11)
    full, single, blind = n["arch"]["полная"], n["arch"]["одноагентная"], n["arch"]["без оптимизатора"]
    d.bullets(s, MARGIN, Inches(4.0), Inches(7.3), Inches(2.6), [
        f"Одноагентная двигает уставки в {single['суммарно °C'] / full['суммарно °C']:.0f} "
        f"раз больше и пропускает не меньше ({single['доля пропусков']:.2f} против "
        f"{full['доля пропусков']:.2f})",
        f"Без оптимизатора система почти не действует: пропусков {blind['доля пропусков']:.2f}",
        "Без агента надёжности — больше ложных тревог и хода уставок",
    ], size=15)
    d.text(s, MARGIN, Inches(6.55), W - 2 * MARGIN, Inches(0.4),
           [("docs/HARD_CHECKS.md §7 · reports/architectures.json · "
             "scripts/compare_architectures.py", 12, False, GRAY)])


def card_slide(d: Deck):
    cmd, block = card_example()
    s = d.slide("Карточка оператора", notes=(
        "Каждая строка карточки и её источник — docs/CARD.md. Строки «Достоверность», "
        "«Разбор причины» и «Возврат в норму» появляются, только когда им есть что "
        "сказать."))
    lines = [(line, 12.5, False, INK, MONO) for line in block.splitlines()]
    height = Inches(min(5.0, 0.235 * len(lines) + 0.4))     # рамка по тексту
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, MARGIN, Inches(1.45),
                             W - 2 * MARGIN, height)
    box.adjustments[0] = 0.02
    d.fill(box, LIGHT)
    d.text(s, MARGIN + Inches(0.2), Inches(1.55), W - 2 * MARGIN - Inches(0.4),
           Inches(4.8), lines)
    d.text(s, MARGIN, Inches(6.55), W - 2 * MARGIN, Inches(0.4),
           [(f"{cmd} · docs/CARD.md", 12, False, GRAY)])


def refusals_slide(d: Deck, n: dict):
    s = d.slide("Отказ — это решение, и он стоит там, где надо", notes=(
        "Отказ по данным — там, где прибор врёт: оперативное значение ошибается в разы "
        "сильнее, а доля превышений в отказах не ниже обычной — отказы не прячут "
        "превышения."))
    ref = n["refusals"]
    decided = ref["с решением"]
    rows = [["Исход на тесте", "Доля моментов", "Превышений за 24 ч",
             "Ошибка оперативного значения, мг/кг"]]
    for key, label in [("с решением", "решение (действие или держим)"),
                       ("отказ: установка остановлена", "отказ: установка стоит"),
                       ("отказ: данные недостоверны", "отказ: данные недостоверны"),
                       ("отказ: нет допустимых вариантов", "отказ: нет допустимых вариантов")]:
        row = ref.get(key) or {}
        if not row:
            continue
        over = row.get("превышений за 24 ч")
        rows.append([label, f"{100 * row['доля моментов']:.1f} %",
                     "—" if over is None else f"{100 * over:.1f} %",
                     f"{row['ошибка оперативного значения, мг/кг']:.2f}"])
    d.table(s, MARGIN, Inches(1.5), W - 2 * MARGIN, rows, [4.2, 2.2, 2.6, 3.13],
            size=13, row_h=0.45)
    bad = ref.get("отказ: данные недостоверны") or {}
    ratio = (bad.get("ошибка оперативного значения, мг/кг", 0)
             / decided["ошибка оперативного значения, мг/кг"])
    d.bullets(s, MARGIN, Inches(4.1), W - 2 * MARGIN, Inches(2.4), [
        f"В отказах по недостоверным данным прибор расходится с лабораторией в "
        f"{ratio:.1f} раза сильнее, чем в моментах с решением",
        hidden_line(ref, report("refusal_quality.json") or {}),
        no_options_line(ref),
    ], size=16)
    d.text(s, MARGIN, Inches(6.55), W - 2 * MARGIN, Inches(0.4),
           [("docs/CARD.md → отказы · reports/refusal_quality.json", 12, False, GRAY)])


def hidden_line(ref: dict, check: dict) -> str:
    """Не прячут ли отказы превышения — по правилу проверки, как оно посчитано."""
    bad = 100 * ((ref.get("отказ: данные недостоверны") or {}).get("превышений за 24 ч") or 0)
    decided = 100 * (ref.get("с решением") or {}).get("превышений за 24 ч", 0)
    ok = (check.get("правило") or {}).get("2. отказы не прячут превышения")
    if ok:
        return (f"Превышений в них {bad:.1f} % против {decided:.1f} % в моментах с "
                "решением: отказ не прячет трудные моменты")
    return (f"Превышений в них {bad:.1f} % против {decided:.1f} % — больше допуска нашего "
            "правила (5 п.п.): после вета тяжести в решения перешли спокойные моменты "
            "конца цикла, база сравнения сместилась. Говорим это сами")


def no_options_line(ref: dict) -> str:
    """«Нет допустимых вариантов» — строка по вердикту проверки вето тяжести."""
    share = 100 * (ref.get("отказ: нет допустимых вариантов") or {}).get("доля моментов", 0)
    veto = report("severity_veto.json")
    if veto is None:
        return (f"«Нет допустимых вариантов» — {share:.1f} % моментов теста, в основном "
                "конец цикла катализатора: высокая тяжесть вычёркивает все варианты. "
                "Найдено 21.09, кандидат проверяется по правилу до счёта")
    if veto.get("принят"):
        before = veto.get("тест", {}).get("all", {})
        was = before.get("нет допустимых вариантов")
        total = before.get("моментов")
        was_txt = f" (было {100 * was / total:.1f} %)" if was and total else ""
        return (f"«Нет допустимых вариантов» — {share:.1f} % моментов теста{was_txt}: "
                "высокая тяжесть режима вычёркивает теперь только варианты, которые греют "
                "или утяжеляют режим — по правилу до счёта")
    return (f"«Нет допустимых вариантов» — {share:.1f} % моментов теста, в основном конец "
            "цикла катализатора; мягкое вето тяжести проверено и отклонено — "
            "reports/severity_veto.json")


def decisions_slide(d: Deck):
    accepted, refused = verdicts()
    s = d.slide("Решения по правилам, записанным до счёта", notes=(
        "Правило приёмки пишется в docs/PLAN.md до того, как посчитан результат, и "
        "исполняется как есть. Отклонённое записывается с числами так же подробно, как "
        "принятое. Вердикты на слайде читаются из отчётов проверок."))
    col = (W - 2 * MARGIN - Inches(0.4)) / 2
    for i, (head, items, color) in enumerate([
            (f"Включено ({len(accepted)})", accepted, GREEN),
            (f"Проверено и отклонено ({len(refused)})", refused, RED)]):
        x = MARGIN + (col + Inches(0.4)) * i
        d.text(s, x, Inches(1.4), col, Inches(0.5), [(head, 18, True, color)])
        size = 13 if len(items) <= 12 else 12
        d.text(s, x, Inches(1.95), col, Inches(4.6),
               [("•  " + label, size, False, INK) for label, _ in items], spacing=3)
    d.text(s, MARGIN, Inches(6.55), W - 2 * MARGIN, Inches(0.4),
           [("docs/PLAN.md — журнал решений · правило каждой проверки — в докстринге "
             "её скрипта в scripts/", 12, False, GRAY)])


def limits_slide(d: Deck, n: dict):
    s = d.slide("Ограничения — сказанные вслух", notes=(
        "Говорить до того, как спросят. У каждого пункта — чем измерено."))
    d.bullets(s, MARGIN, Inches(1.5), W - 2 * MARGIN, Inches(5.0), [
        "За сутки вперёд система не предупреждает: она реагирует на уже идущее "
        f"превышение; прогноз на 2 часа у бустинга — ROC-AUC {fmt(n['auc_h2'])}",
        "Отклик серы на режим взят из практики технолога (0.3–1.0 мг/кг на градус) и "
        "кинетики, а не измерен на истории — там его гасят регуляторы",
        "Рабочие диапазоны уставок — квантили истории: паспортных ограничений в пакете "
        "нет, эксперт назвал их конфиденциальными",
        "Т95 — ограничение «не ухудшать», а не гарантия: MAE оценок около 5 °C при "
        "запасе до предела 3–5 °C",
        "Управляющие воздействия — только гидроочистки; связь АВТ с гидроочисткой "
        "измерена и оказалась слабой",
        "Смешение — модель на явных допущениях: данных по компонентам не выдано",
    ], size=17, gap=12)


def check_slide(d: Deck):
    s = d.slide("Как проверить за 15 минут", notes=(
        "README — входная точка: таблица «за пять минут» ведёт к каждому числу и "
        "документу. Всё работает без интернета."))
    cmds = [("python scripts/prepare_data.py", "распаковка и кэш, один раз"),
            ("python scripts/run_cycle.py --ts \"2026-03-05 00:00\"", "одна карточка целиком"),
            ("streamlit run app/dashboard.py", "дашборд: сцены защиты, путь решения"),
            ("python scripts/run_scenario.py --list", "сценарии с возмущениями"),
            ("pytest -q", "тесты, включая сверку документов с отчётами"),
            ("bash scripts/reproduce_all.sh", "все отчёты заново, с продолжением с места")]
    y = Inches(1.5)
    for cmd, what in cmds:
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, MARGIN, y, Inches(7.3),
                                 Inches(0.55))
        box.adjustments[0] = 0.15
        d.fill(box, LIGHT)
        d.text(s, MARGIN + Inches(0.15), y + Inches(0.08), Inches(7.0), Inches(0.4),
               [(cmd, 14, False, NAVY, MONO)])
        d.text(s, Inches(8.1), y + Inches(0.08), Inches(4.6), Inches(0.4),
               [(what, 14, False, INK)])
        y += Inches(0.7)
    d.text(s, MARGIN, Inches(5.9), W - 2 * MARGIN, Inches(0.6),
           [("Куда смотреть: README → «За пять минут» · docs/CARD.md · "
             "docs/HARD_CHECKS.md · docs/DATA_PERIODS.md · docs/PLAN.md", 14, False, GRAY)])


def main() -> int:
    use_utf8_console()
    n = numbers()
    deck = Deck()
    title_slide(deck)
    what_slide(deck)
    architecture_slide(deck)
    data_slide(deck, n)
    numbers_slide(deck, n)
    quality_slide(deck, n)
    reliability_slide(deck, n)
    optimizer_slide(deck, n)
    architectures_slide(deck, n)
    card_slide(deck)
    refusals_slide(deck, n)
    decisions_slide(deck)
    limits_slide(deck, n)
    check_slide(deck)
    deck.save(OUT)
    accepted, refused = verdicts()
    print(f"Слайдов: {deck.n}; решений включено {len(accepted)}, отклонено {len(refused)}")
    print(f"Презентация: {OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
