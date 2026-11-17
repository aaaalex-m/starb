r"""Lint maths in Markdown so that it renders on GitHub.

GitHub renders ``$...$`` (inline), ``$$...$$`` on their own lines, and fenced ``math`` blocks
(display) with MathJax, but parses Markdown *first*, which breaks some LaTeX. This checks the
repository's Markdown files and the Markdown cells of the jupytext notebooks for:

* ``\(...\)`` or ``\[...\]`` delimiters (not supported by GitHub);
* ``$$`` sharing a line with other text;
* inside inline or ``$$`` maths: ``<``/``>`` (use ``\lt``/``\gt``); a backslash before
  punctuation, which Markdown treats as an escape (``\,`` becomes ``,``; use ``\thinspace``,
  ``\lbrace``/``\rbrace`` for ``\{``/``\}``); ``*`` (use ``\ast``); and ``_`` right after
  ``}``, ``)``, ``]``, ``|`` or a space (read as emphasis; reorder, e.g. ``P_t^{x}``);
* a closing ``$`` followed by a letter or digit (``$k$th``) or by ``)`` after maths that
  ends in ``)``;
* ``|`` inside inline maths in a table row (use ``\lvert``/``\rvert``);
* maths in headings or inside single-``*``/``_`` emphasis.

Fenced ``math`` blocks are code to Markdown, so their content is not checked. Usage:
``python scripts/check_math.py [files...]`` (default: all tracked Markdown and notebook
sources); exits 1 if anything is flagged.
"""

import re
import subprocess
import sys
from pathlib import Path

INLINE = re.compile(r"(?<![\\$])\$(?![\s$])((?:[^$\n\\]|\\.)+?)(?<!\s)\$(?!\$)")
DISPLAY = re.compile(r"(?<!\\)\$\$(.+?)\$\$", re.S)
CODE_SPAN = re.compile(r"`[^`\n]*`")
EMPHASIS_WITH_MATH = re.compile(
    r"(?<![*\w\\])([*_])(?![*_\s])[^*_\n]*?\$[^$\n]+\$[^*_\n]*?(?<![\s*_])\1(?![*_\w])"
)

MATH_RULES = [
    (re.compile(r"[<>]"), "use \\lt / \\gt instead of < / >"),
    (
        re.compile(r"""\\[!"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~\\]"""),
        "Markdown eats a backslash before punctuation (\\, \\; \\{ \\} \\\\ ...): use "
        "\\thinspace, \\lbrace / \\rbrace, \\Vert, or a fenced math block",
    ),
    (re.compile(r"\*"), "use \\ast instead of *"),
    (re.compile(r"[})\]|\s]_"), "'_' after '}', ')', ']', '|' or a space is read as emphasis"),
]


def markdown_chunks(path: Path) -> list[tuple[int, str]]:
    """(first line number, text) of each Markdown paragraph, with code removed."""
    lines = path.read_text().split("\n")
    if path.suffix == ".py":
        # jupytext percent format: Markdown cells are comment lines after "# %% [markdown]"
        md, in_md = [], False
        for line in lines:
            if line.startswith("# %%"):
                in_md = "[markdown]" in line
                md.append("")
            elif in_md and (line.startswith("# ") or line == "#"):
                md.append(line[2:])
            else:
                in_md = False if not line.startswith("#") else in_md
                md.append("")
        lines = md

    chunks, current, start, fence = [], [], 0, None
    for n, line in enumerate(lines, 1):
        marker = re.match(r"^\s*(```+|~~~+)", line)
        if fence:
            if marker and marker.group(1).startswith(fence):
                fence = None
            continue
        if marker:
            fence = marker.group(1)
            if current:
                chunks.append((start, "\n".join(current)))
                current = []
            continue
        if line.strip():
            if not current:
                start = n
            current.append(CODE_SPAN.sub(lambda m: " " * len(m.group()), line))
        elif current:
            chunks.append((start, "\n".join(current)))
            current = []
    if current:
        chunks.append((start, "\n".join(current)))
    return chunks


def check_chunk(start: int, text: str) -> list[tuple[int, str]]:
    problems = []

    def line_of(pos: int) -> int:
        return start + text[:pos].count("\n")

    for m in re.finditer(r"\\[(\[]", text):
        problems.append((line_of(m.start()), "use $...$ / $$...$$ instead of \\( or \\["))
    for m in re.finditer(r"^.*\$\$.*$", text, re.M):
        stripped = m.group().strip()
        if stripped != "$$" and not re.fullmatch(r"\$\$[^$]+\$\$", stripped):
            problems.append((line_of(m.start()), "display maths ($$) must be alone on its line"))

    display_spans = [m.span() for m in DISPLAY.finditer(text)]
    for m in DISPLAY.finditer(text):
        for rule, message in MATH_RULES[:3]:
            if rule.search(m.group(1)):
                problems.append((line_of(m.start()), f"{message}: $${m.group(1).strip()[:60]}$$"))

    for m in INLINE.finditer(text):
        if any(a <= m.start() < b for a, b in display_spans):
            continue
        latex, line = m.group(1), line_of(m.start())
        for rule, message in MATH_RULES:
            if rule.search(latex):
                problems.append((line, f"{message}: ${latex}$"))
        row = text[text.rfind("\n", 0, m.start()) + 1 :]
        if row.lstrip().startswith("|") and "|" in latex:
            problems.append((line, f"'|' splits table cells; use \\lvert / \\rvert: ${latex}$"))
        after = text[m.end() : m.end() + 1]
        if after.isalnum():
            problems.append((line, f"closing $ followed by '{after}': ${latex}$"))
        if after == ")" and latex.endswith(")"):
            problems.append((line, f"')$)' is not recognised as maths: ${latex}$"))

    for m in re.finditer(r"^#{1,6}\s.*\$.*$", text, re.M):
        problems.append((line_of(m.start()), "no maths in headings"))
    for m in EMPHASIS_WITH_MATH.finditer(text):
        problems.append((line_of(m.start()), "no maths inside single */_ emphasis"))
    return problems


def default_files() -> list[Path]:
    tracked = subprocess.run(
        ["git", "ls-files", "*.md", "modules/*/notebook.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [Path(f) for f in tracked]


def main(argv: list[str]) -> int:
    files = [Path(a) for a in argv] or default_files()
    count = 0
    for path in files:
        if path.suffix not in {".md", ".py"}:
            continue
        for start, text in markdown_chunks(path):
            for line, message in check_chunk(start, text):
                print(f"{path}:{line}: {message}")
                count += 1
    if count:
        print(f"{count} maths formatting problem(s); see scripts/check_math.py for the rules")
    return 1 if count else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
