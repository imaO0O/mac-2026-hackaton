"""Страницы HTML для архива: ссылки и якоря работают без GitHub.

Архив для письма читают в браузере (README.html), а не на GitHub. Ссылка, живая в
.md, могла бы умереть при переводе: .md → .html, якорь заголовка, схема картинкой
вместо mermaid. Проверяется то, что увидит проверяющий: каждая относительная ссылка
ведёт на страницу или на файл, который есть в архиве, а якорь — на заголовок.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("markdown")

from nefte.config import ROOT  # noqa: E402

HREF = re.compile(r'(?:href|src)="([^"]+)"')
ID = re.compile(r'id="([^"]+)"')


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    sys.path.insert(0, str(ROOT))
    from scripts.make_html_docs import build

    out = tmp_path_factory.mktemp("html")
    return out, build(out)


def test_every_link_and_anchor_resolves(pages):
    import html

    sys.path.insert(0, str(ROOT))
    from scripts.make_submission import collect

    out, written = pages
    shipped = {p.relative_to(ROOT).as_posix() for p in collect(with_reports=True)}
    shipped |= {p.relative_to(out).as_posix() for p in written}
    ids = {p.relative_to(out).as_posix(): set(ID.findall(p.read_text(encoding="utf-8")))
           for p in written}
    broken = []
    for page in written:
        here = page.relative_to(out).parent
        for target in HREF.findall(page.read_text(encoding="utf-8")):
            target = html.unescape(target)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            file_part, _, anchor = target.partition("#")
            rel = (here / file_part).as_posix() if file_part else page.relative_to(out).as_posix()
            rel = Path(rel).as_posix()
            parts = []
            for piece in rel.split("/"):
                if piece == "..":
                    parts.pop()
                elif piece not in ("", "."):
                    parts.append(piece)
            rel = "/".join(parts)
            exists = rel in shipped or any(name.startswith(rel.rstrip("/") + "/") for name in shipped)
            if not exists:
                broken.append(f"{page.name} → {target}")
            elif anchor and rel.endswith(".html") and anchor not in ids.get(rel, set()):
                broken.append(f"{page.name} → {target}: нет заголовка")
    assert not broken, "; ".join(broken[:10])


def test_readme_page_shows_the_diagram_instead_of_mermaid(pages):
    out, _ = pages
    text = (out / "README.html").read_text(encoding="utf-8")
    assert "flowchart" not in text and 'src="docs/architecture.png"' in text
