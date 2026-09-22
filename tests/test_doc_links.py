"""Все ссылки README и документов ведут туда, куда обещают.

Совет экспертов (21.09): решение открывают впервые, может быть девяносто пятым за
день, и если по README не видно, где результаты и что смотреть, его не станут
разбирать. Просьба — «грамотный README… прощёлкайте все ссылки». Прощёлкать один раз
мало: следующая же правка документа молча ломает якорь. Поэтому ссылки проверяются
тестом: файл существует, а якорь `#…` указывает на заголовок, который действительно
есть в целевом файле, — якорь строится тем же правилом, что у GitHub.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from nefte.config import ROOT

LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$", re.M)
FENCE = re.compile(r"^```.*?^```", re.M | re.S)

DOCUMENTS = sorted([ROOT / "README.md", *(ROOT / "docs").glob("*.md")])


def github_slug(text: str) -> str:
    """Якорь заголовка так, как его строит GitHub: строчные, без знаков, пробел → «-»."""
    text = re.sub(r"`|\*\*|__", "", text.strip().lower())
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)      # ссылка в заголовке
    text = re.sub(r"[^\w\- ]", "", text)                       # \w — с кириллицей
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    seen: dict[str, int] = {}
    out = set()
    for _, title in HEADING.findall(text):
        slug = github_slug(title)
        count = seen.get(slug, 0)
        out.add(slug if count == 0 else f"{slug}-{count}")
        seen[slug] = count + 1
    return out


def links(path: Path) -> list[str]:
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    return [target for target in LINK.findall(text)
            if not target.startswith(("http://", "https://", "mailto:"))]


def test_slug_matches_github_on_our_headings():
    assert github_slug("2. Ловушки (главное)") == "2-ловушки-главное"
    assert github_slug("1. «Злые» кейсы") == "1-злые-кейсы"
    assert github_slug("Что происходит, когда система отказывается") == \
        "что-происходит-когда-система-отказывается"
    assert github_slug("Карточка — пример") == "карточка--пример"


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: p.name)
def test_every_relative_link_resolves(document: Path):
    broken = []
    for target in links(document):
        file_part, _, anchor = target.partition("#")
        destination = (document.parent / file_part).resolve() if file_part else document
        if not destination.exists():
            broken.append(f"{target}: нет файла")
            continue
        if anchor and destination.suffix == ".md":
            if anchor.lower() not in anchors(destination):
                broken.append(f"{target}: нет заголовка «{anchor}»")
    assert not broken, f"{document.name}: " + "; ".join(broken)


def test_readme_actually_links_what_it_mentions():
    """README не должен называть файлы, не давая по ним перейти.

    Раньше все пути в README стояли в обратных кавычках — прочитать можно,
    щёлкнуть нельзя. Разрешены только команды и имена внутри блоков кода.
    """
    text = FENCE.sub("", (ROOT / "README.md").read_text(encoding="utf-8"))
    linked = set(links(ROOT / "README.md"))
    mentioned = set(re.findall(r"`((?:docs|reports|app|scripts|src|configs|tests)/[^`\s]+?\.(?:md|json|py|yaml))`", text))
    unlinked = sorted(path for path in mentioned
                      if not any(target.split("#")[0] == path for target in linked))
    assert not unlinked, f"упомянуты без ссылки: {unlinked}"


def test_every_link_ships_in_the_submission_archive():
    """Проверяющий откроет не репозиторий, а архив. Ссылка, которая жива в
    репозитории, но ведёт на файл, не попавший в архив, для него битая: так было с
    `.env.example` — README ссылался на образец, а архив его не брал."""
    import sys

    sys.path.insert(0, str(ROOT))
    from scripts.make_submission import collect

    shipped = {p.relative_to(ROOT).as_posix() for p in collect(with_reports=True)}
    missing = []
    for document in DOCUMENTS:
        for target in links(document):
            file_part = target.partition("#")[0]
            if not file_part:
                continue
            destination = (document.parent / file_part).resolve()
            if not destination.is_relative_to(ROOT.resolve()) or not destination.exists():
                continue                    # битые ссылки ловит тест выше
            rel = destination.relative_to(ROOT.resolve()).as_posix()
            if destination.is_dir():
                ok = any(name.startswith(rel.rstrip("/") + "/") for name in shipped)
            else:
                ok = rel in shipped
            if not ok:
                missing.append(f"{document.name} → {target}")
    assert not missing, "не попадут в архив: " + "; ".join(sorted(set(missing)))
