#!/usr/bin/env bash
# Typeset a module's lecture notes as a PDF with pandoc + XeLaTeX.
#
#   scripts/lecture_pdf.sh modules/01-returns/lecture.md out/01-returns.pdf
#
# Needs pandoc (>= 3) and a TeX Live with xelatex, unicode-math, tcolorbox and the TeX Gyre
# fonts (Ubuntu: pandoc texlive-xetex texlive-latex-extra texlive-fonts-recommended).
# GitHub-only syntax is translated for print: alerts (> [!NOTE], [!TIP], [!WARNING]) become
# coloured boxes, <details> blocks are printed expanded under an "Optional" title, fenced
# math blocks become display maths, and "**Figure N.**" captions become LaTeX figures.
# PDFs are build products: don't commit them.
set -euo pipefail

src=$1
out=$2
module_dir=$(dirname "$src")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

python3 - "$src" "$module_dir" > "$tmp/notes.md" <<'EOF'
import posixpath
import re
import sys

src, module_dir = sys.argv[1], sys.argv[2]
text = open(src).read()

# Title from the first heading; the remaining headings move up one level.
title, text = re.match(r"#\s+(.+?)\n(.*)", text, re.S).groups()

# GitHub display maths (fenced math blocks) -> pandoc display maths.
text = re.sub(r"```math\n(.*?)\n```", lambda m: f"$$\n{m.group(1)}\n$$", text, flags=re.S)

# On paper nothing is collapsed: reword the reading guide.
text = text.replace("collapsed sections (click to expand)", "optional sections (printed in full between grey rules)")

# GitHub alerts -> fenced divs (styled by the Lua filter).
def alert(m):
    body = re.sub(r"^>[ ]?", "", m.group(2), flags=re.M)
    return f"::: {m.group(1).lower()}\n{body}\n:::\n"

text = re.sub(r"^> \[!(NOTE|TIP|WARNING|IMPORTANT|CAUTION)\]\n((?:>.*\n?)*)", alert, text, flags=re.M)

# <details>/<summary> -> fenced divs, expanded. Top-level ones get an "Optional" title.
out_lines, depth = [], 0
for line in text.split("\n"):
    opening = re.match(r"\s*<details>\s*(?:<summary>(.*?)</summary>)?\s*(.*)$", line)
    if opening:
        depth += 1
        out_lines.append("")
        out_lines.append("::: details" if depth == 1 else "::: subdetails")
        rest = opening.group(2)
        if opening.group(1) is not None:
            out_lines.append("")
            label = opening.group(1)
            out_lines.append(f"**Optional: {label}**" if depth == 1 else f"**{label}.**")
            if rest:
                out_lines.append(rest)
        continue
    summary = re.match(r"\s*<summary>(.*?)</summary>\s*$", line)
    if summary:
        label = summary.group(1)
        out_lines.append("")
        out_lines.append(f"**Optional: {label}**" if depth == 1 else f"**{label}.**")
        out_lines.append("")
        continue
    if re.match(r"\s*</details>\s*$", line):
        depth -= 1
        out_lines += ["", ":::", ""]
        continue
    out_lines.append(line)
text = "\n".join(out_lines)

# An image followed by a "**Figure N.** caption" paragraph -> one LaTeX figure with that
# caption (LaTeX numbers it); image paths stay relative to the module folder.
text = re.sub(
    r"!\[[^\]]*\]\(([^)\s]+)\)\n\n\*\*Figure \d+\.\*\* (.+)",
    lambda m: f"![{m.group(2)}]({m.group(1)}){{width=100%}}",
    text,
)

# Relative links (not images) -> GitHub, so they work from the PDF.
def absolute(m):
    label, target = m.group(1), m.group(2)
    if re.match(r"[a-z]+:", target) or target.startswith("#"):
        return m.group(0)
    path = posixpath.normpath(posixpath.join(module_dir, target))
    return f"[{label}](https://github.com/aaaalex-m/starb/blob/main/{path})"

text = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", absolute, text)
title = title.replace('"', '\\"')
print(f'---\ntitle: "{title}"\nsubtitle: "starb: learning statistical arbitrage in Python"\n---\n')
print(text)
EOF

cat > "$tmp/callouts.lua" <<'EOF'
local envs = {
  note = "notebox", tip = "tipbox", important = "notebox",
  warning = "warningbox", caution = "warningbox", details = "detailsbox",
}
function Div(el)
  local class = el.classes[1]
  if class == "subdetails" then
    return el.content
  end
  local env = envs[class]
  if env then
    local blocks = {pandoc.RawBlock("latex", "\\begin{" .. env .. "}")}
    for _, block in ipairs(el.content) do
      table.insert(blocks, block)
    end
    table.insert(blocks, pandoc.RawBlock("latex", "\\end{" .. env .. "}"))
    return blocks
  end
end
EOF

cat > "$tmp/header.tex" <<'EOF'
\providecommand{\lt}{<}
\providecommand{\gt}{>}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small}
\setlength{\emergencystretch}{3em}
\usepackage{float}
\floatplacement{figure}{H}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[most]{tcolorbox}
\tcbset{callout/.style={enhanced, breakable, boxrule=0pt, leftrule=2.5pt, arc=0pt,
  outer arc=0pt, left=6pt, right=6pt, top=4pt, bottom=4pt, fonttitle=\bfseries,
  coltitle=black, attach title to upper, after title={.\enspace}, before skip=8pt,
  after skip=8pt}}
\newtcolorbox{notebox}{callout, colback=blue!4, colframe=blue!55!black, title={Note}}
\newtcolorbox{tipbox}{callout, colback=green!5, colframe=green!45!black, title={Tip}}
\newtcolorbox{warningbox}{callout, colback=orange!7, colframe=orange!85!black, title={Warning}}
\definecolor{detailsrule}{gray}{0.6}
\newenvironment{detailsbox}
  {\par\medskip\noindent{\color{detailsrule}\rule{\linewidth}{0.8pt}}\par\nopagebreak}
  {\par\nopagebreak\noindent{\color{detailsrule}\rule{\linewidth}{0.8pt}}\par\medskip}
EOF

pandoc "$tmp/notes.md" \
  --from markdown+tex_math_dollars+pipe_tables+raw_html+fenced_divs \
  --shift-heading-level-by=-1 \
  --lua-filter="$tmp/callouts.lua" \
  --pdf-engine=xelatex \
  --include-in-header="$tmp/header.tex" \
  --resource-path="$module_dir" \
  -V documentclass=article -V fontsize=11pt -V geometry:margin=2.4cm \
  -V mainfont="TeX Gyre Pagella" -V mathfont="TeX Gyre Pagella Math" \
  -V monofont="DejaVu Sans Mono" -V monofontoptions="Scale=0.85" \
  -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
  -V date="$(date +%Y-%m-%d)" \
  --highlight-style=tango \
  -o "$out"
