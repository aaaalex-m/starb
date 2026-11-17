#!/usr/bin/env bash
# Typeset a module's lecture notes as a PDF with pandoc + XeLaTeX.
#
#   scripts/lecture_pdf.sh modules/01-returns/lecture.md out/01-returns.pdf
#
# Needs pandoc (>= 3) and a TeX Live with xelatex, unicode-math and the TeX Gyre fonts
# (Ubuntu: pandoc texlive-xetex texlive-latex-extra texlive-fonts-recommended).
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

# Collapsible interview answers -> a visible "Key points" paragraph.
text = re.sub(r"<details><summary>(.*?)</summary>\s*", r"**\1.** ", text)
text = text.replace("</details>", "")

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

cat > "$tmp/header.tex" <<'EOF'
\providecommand{\lt}{<}
\providecommand{\gt}{>}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small}
\setlength{\emergencystretch}{3em}
\usepackage{float}
\floatplacement{figure}{H}
\usepackage[font=small,labelfont=bf]{caption}
EOF

pandoc "$tmp/notes.md" \
  --from markdown+tex_math_dollars+pipe_tables+raw_html \
  --shift-heading-level-by=-1 \
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
