"""README и документы — страницами HTML, чтобы читать их как на GitHub, без него.

    pip install markdown                       # только для этого скрипта
    python scripts/make_html_docs.py --out <папка>

Зачем. Решение уходит архивом, а README.md, открытый двойным щелчком в Windows, —
это Блокнот с вертикальными чертами вместо таблиц. Здесь каждый .md становится
страницей рядом с тем же путём (README.md → README.html, docs/CARD.md →
docs/CARD.html): таблицы, код, схема агентов картинкой, ссылки между документами
ведут на страницы, а на отчёты и скрипты — как есть. Якоря заголовков строятся тем
же правилом, что у GitHub (`tests/test_doc_links.py`), поэтому ссылки вида
`CARD.md#что-происходит-…` работают и здесь.

Страницы не хранятся в репозитории: их собирает `scripts/make_submission.py` при
упаковке архива.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import markdown

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nefte.config import ROOT  # noqa: E402
from nefte.utils import use_utf8_console  # noqa: E402

MERMAID = re.compile(r"^```mermaid\n.*?^```\n", re.M | re.S)
MD_LINK = re.compile(r"(\]\()(?!https?:|mailto:)([^)#\s]+?)\.md((?:#[^)\s]*)?\))")
HEADING = re.compile(r"<h([1-6])>(.*?)</h\1>", re.S)
TAG = re.compile(r"<[^>]+>")

CSS = """
:root { color-scheme: light; }
body { margin: 0; background: #ffffff; color: #1f2328;
  font: 16px/1.6 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
.top { background: #f6f8fa; border-bottom: 1px solid #d1d9e0; padding: 10px 16px; font-size: 14px; }
.top a { color: #0969da; text-decoration: none; margin-right: 16px; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 64px; }
h1, h2 { border-bottom: 1px solid #d1d9e0; padding-bottom: .3em; }
h1 { font-size: 2em; } h2 { font-size: 1.5em; margin-top: 1.6em; } h3 { margin-top: 1.4em; }
a { color: #0969da; } a:hover { text-decoration: underline; }
code { background: #eff1f3; border-radius: 6px; padding: .15em .35em; font-size: 85%;
  font-family: Consolas, "SFMono-Regular", Menlo, monospace; }
pre { background: #f6f8fa; border-radius: 6px; padding: 14px; overflow: auto; line-height: 1.45; }
pre code { background: none; padding: 0; font-size: 13px; }
table { border-collapse: collapse; display: block; overflow: auto; margin: 1em 0; }
th, td { border: 1px solid #d1d9e0; padding: 6px 13px; vertical-align: top; }
th { background: #f6f8fa; font-weight: 600; } tr:nth-child(2n) td { background: #f6f8fa; }
blockquote { margin: 0; padding: 0 1em; color: #59636e; border-left: .25em solid #d1d9e0; }
img { max-width: 100%; }
"""


def github_slug(text: str) -> str:
    """То же правило, что в tests/test_doc_links.py — якоря совпадают с GitHub."""
    text = re.sub(r"`|\*\*|__", "", text.strip().lower())
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def with_anchors(body: str) -> str:
    seen: dict[str, int] = {}

    def repl(match: re.Match) -> str:
        level, inner = match.group(1), match.group(2)
        slug = github_slug(html.unescape(TAG.sub("", inner)))
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchor = slug if count == 0 else f"{slug}-{count}"
        return f'<h{level} id="{html.escape(anchor, quote=True)}">{inner}</h{level}>'

    return HEADING.sub(repl, body)


def render(source: Path, root: Path) -> str:
    text = source.read_text(encoding="utf-8")
    rel_root = Path(*([".."] * (len(source.relative_to(root).parts) - 1))) \
        if len(source.relative_to(root).parts) > 1 else Path(".")
    picture = (rel_root / "docs" / "architecture.png").as_posix()
    # GitHub рисует mermaid сам; браузер без интернета — нет: схема картинкой
    text = MERMAID.sub(f"![Схема агентов и обмена между ними]({picture})\n", text)
    text = MD_LINK.sub(lambda m: f"{m.group(1)}{m.group(2)}.html{m.group(3)}", text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    body = with_anchors(body)
    title = html.escape(TAG.sub("", HEADING.search(body).group(2)) if HEADING.search(body)
                        else source.stem)
    readme = (rel_root / "README.html").as_posix()
    nav = (f'<div class="top"><a href="{readme}">← README</a>'
           f'<span>{html.escape(source.relative_to(root).as_posix())}</span></div>')
    return (f"<!doctype html>\n<html lang=\"ru\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{title}</title><style>{CSS}</style></head>"
            f"<body>{nav}<main>{body}</main></body></html>\n")


def sources(root: Path) -> list[Path]:
    return [root / "README.md", *sorted((root / "docs").glob("*.md"))]


def build(out: Path, root: Path = ROOT) -> list[Path]:
    written = []
    for source in sources(root):
        target = out / source.relative_to(root).with_suffix(".html")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(source, root), encoding="utf-8")
        written.append(target)
    return written


def main() -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "html_preview")
    args = parser.parse_args()
    written = build(args.out)
    print(f"страниц: {len(written)} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
