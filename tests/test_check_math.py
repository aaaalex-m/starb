"""The maths lint flags exactly the patterns GitHub's renderer was observed to break."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "check_math", Path(__file__).parents[1] / "scripts" / "check_math.py"
)
check_math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_math)

RENDERS = [
    r"Inline $x_t$ and $P_t^{\text{adj}} / P_{t-1}^{\text{adj}}$ maths.",
    r"Comparisons $\lvert\phi\rvert \lt 1$ and $a \gt b$, sets $\lbrace 1, 2 \rbrace$.",
    r"Punctuation after maths ($x$), $I(1)$. and the $t$-statistic.",
    r"Bold maths **$\beta$** is fine.",
    "Display maths:\n$$\n\\operatorname{Var}(x) = \\sigma^2\n$$",
    "$$\\ln\\frac{P_T}{P_0} = \\sum_t r_t$$",
]

BREAKS = [
    r"Delimiters \(x\) are not supported.",
    r"Display \[x\] is not supported.",
    r"Less-than $r < R$ is double-escaped.",
    r"Escapes $\mu\,dt$ lose their backslash.",
    r"Braces $\{a\}$ lose their backslash.",
    r"Emphasis $P^{a}_t / P^{b}_{t-1}$ breaks.",
    r"Stars $x^*$ break.",
    r"Parenthesis (a random walk, $I(1)$) is not maths.",
    r"Letters after $k$th are not maths.",
    r"Italic *assuming $H_0$ is true* is not maths.",
    "Inline maths $a +\nb$ across lines.",
    "Text before $$x$$ on the same line.",
    r"| table | $\|x\| + |y|$ |",
    r"## Heading with $x$",
]


@pytest.mark.parametrize("text", RENDERS)
def test_renderable_maths_passes(text):
    assert check_math.check_chunk(1, text) == []


@pytest.mark.parametrize("text", BREAKS)
def test_broken_maths_is_flagged(text):
    assert check_math.check_chunk(1, text)


def test_code_spans_and_fenced_math_are_not_checked(tmp_path):
    md = tmp_path / "notes.md"
    md.write_text("Code `\\(x\\)` and `$a < b$`.\n\n```math\na < b, \\{x\\}, \\\\\n```\n")
    assert check_math.main([str(md)]) == 0


def test_notebook_markdown_cells_are_checked(tmp_path):
    nb = tmp_path / "notebook.py"
    nb.write_text("# %% [markdown]\n# Bad $r < R$ here.\n\n# %%\nx = '$a < b$'\n")
    assert check_math.main([str(nb)]) == 1
    nb.write_text("# %% [markdown]\n# Good $r \\lt R$ here.\n\n# %%\nx = '$a < b$'\n")
    assert check_math.main([str(nb)]) == 0


def test_alerts_and_details(tmp_path):
    md = tmp_path / "notes.md"
    good = (
        "> [!NOTE]\n> Alert maths $a \\lt b$:\n>\n> $$\n> x^2\n> $$\n\n"
        "<details>\n<summary>Derivation of $x$</summary>\n\n"
        "With $x$ inline:\n\n$$y = \\lvert x\\rvert$$\n\n</details>\n"
    )
    md.write_text(good)
    assert check_math.main([str(md)]) == 0
    md.write_text("> [!TIP]\n> Bad alert maths $a < b$.\n")
    assert check_math.main([str(md)]) == 1
    md.write_text("<details>\n<summary>Box</summary>\n\n```math\nx\n```\n\n</details>\n")
    assert check_math.main([str(md)]) == 1
    md.write_text("<details>\n<summary>Box</summary>\n\nWith $x$:\n\n$$\ny\n$$\n\n</details>\n")
    assert check_math.main([str(md)]) == 1
