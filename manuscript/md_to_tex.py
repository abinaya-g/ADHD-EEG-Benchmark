# -*- coding: utf-8 -*-
"""Markdown -> LaTeX (Elsevier cas-sc) converter, purpose-built for manuscript.md.
Not a general converter: handles exactly this file's markdown subset
(headers, pipe tables, bold/italic, numbered/bulleted lists, [N]/[N,M,...]
citation brackets mapped to \\cite{bibkey}, ![caption](path) images mapped to
figure environments) and escapes LaTeX special characters in prose and table
cells. Document-class/frontmatter/backmatter conventions (cas-sc, \\shorttitle,
\\author[n]{}/\\affiliation[n]{}/\\ead/\\cormark, \\begin{abstract}/\\begin{keywords},
CRediT/Ethical statement/\\printcredits backmatter, model1-num-names bib style)
follow the structure of a cas-sc-class reference manuscript supplied by the user.
No LaTeX toolchain is available in this environment to compile and verify the
output; review in Overleaf (or any local TeX install) before submission.
"""
import re

SRC = "/home/user/ADHD-EEG-Benchmark/manuscript/manuscript.md"
OUT = "/home/user/ADHD-EEG-Benchmark/manuscript/manuscript.tex"

NUM_TO_KEY = {
    1: "polanczyk2007worldwide", 2: "thapar2016attention", 3: "faraone2015attention",
    4: "snyder2006metaanalysis", 5: "arns2013decade", 6: "loo2012clinical",
    7: "craik2019deep", 8: "roy2019deep", 9: "lecun2015deep", 10: "lawhern2018eegnet",
    11: "schirrmeister2017deep", 12: "ioffe2015batch", 13: "srivastava2014dropout",
    14: "kingma2015adam", 15: "taghibeyglou2022detection", 16: "amini2025adhdeepnet",
    17: "hassan2024convolutional", 18: "esas2023detection", 19: "chen2019deep",
    20: "latifi2024siamese", 21: "kasim2023identification", 22: "cura2024detection",
    23: "bansal2025eeg", 24: "mao2025advanced", 25: "kim2025electroencephalogram",
    26: "lohani2025systematic", 27: "sanchis2026multiscale", 28: "li2026crosssubject",
    29: "sun2016return", 30: "zhong2023deep", 31: "sundararajan2017axiomatic",
    32: "varma2006bias", 33: "saeb2017need", 34: "vabalas2019machine",
    35: "chicco2020advantages", 36: "efron1979bootstrap", 37: "wilcoxon1945individual",
    38: "holm1979simple", 39: "pedregosa2011scikitlearn", 40: "abadi2016tensorflow",
}


_REGISTRY = {}
_COUNTER = [0]


def escape_text(s):
    """Escape LaTeX special chars in plain prose, then re-render a small set
    of typographic/math tokens this manuscript actually uses.

    Bold/italic spans recurse into this same function (see below), so
    placeholder ids must be globally unique across the whole call tree, not
    just within one invocation -- otherwise a recursive call's own
    from-scratch counter can collide with, and incorrectly resolve, a
    placeholder that belongs to its caller (this was a real bug caught
    during conversion: a backtick span inside an italic run was silently
    replaced by an unrelated math substitution because both used index 0)."""
    def stash(v):
        _COUNTER[0] += 1
        key = _COUNTER[0]
        _REGISTRY[key] = v
        return f"\x00{key}\x00"

    # Citation markers [N] or [N,M,...] -> \cite{...}
    def cite_repl(m):
        nums = [int(x) for x in m.group(1).split(",")]
        keys = ",".join(NUM_TO_KEY[n] for n in nums)
        return stash(f"\\cite{{{keys}}}")
    s = re.sub(r"\[(\d+(?:,\d+)*)\]", cite_repl, s)

    # Inline code spans `like this`
    s = re.sub(r"`([^`]+)`", lambda m: stash("\\texttt{" + tex_escape_basic(m.group(1)) + "}"), s)

    # Bold / italic: recurse through the FULL pipeline (not just basic
    # escaping) so math symbols, citations, etc. inside a bold/italic span
    # still get converted -- e.g. an italic caption containing "20 model
    # x preprocessing" or a citation bracket must not be frozen out of the
    # math/citation passes just because it's nested inside emphasis.
    s = re.sub(r"\*\*(.+?)\*\*", lambda m: stash("\\textbf{" + escape_text(m.group(1)) + "}"), s)
    s = re.sub(r"\*(.+?)\*", lambda m: stash("\\textit{" + escape_text(m.group(1)) + "}"), s)

    # Math / symbol tokens used in this manuscript.
    # Superscript exponent (e.g. "1 × 10⁻⁴" or "1×10⁻⁴") first, regardless
    # of spacing, since the generic "×" rule below would otherwise leave
    # the raw unicode superscript digits/minus unconverted.
    sup_digits = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    sup_to_ascii = str.maketrans(sup_digits, "0123456789")

    def exp_repl(m):
        coeff = m.group(1) or ""
        exp = m.group(2).translate(sup_to_ascii)
        coeff_tex = f"{coeff}\\times" if coeff else ""
        return stash(f"${coeff_tex}10^{{-{exp}}}$")
    s = re.sub(r"(\d+)?\s*×\s*10⁻([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", exp_repl, s)

    s = re.sub(r"128to512", "128-to-512", s)

    math_map = [
        (r"×2", r"$\times$2"),
        (r"×4", r"$\times$4"),
        (r"5×5", r"5$\times$5"),
        (r"α = 0\.05", r"$\alpha$ = 0.05"),
        (r"≤", r"$\leq$"),
        (r"≥", r"$\geq$"),
        (r"×", r"$\times$"),
    ]
    for pat, rep in math_map:
        s = re.sub(pat, lambda m, _r=rep: stash(_r), s)

    s = tex_escape_basic(s)

    # Restore against the GLOBAL registry, not a per-call list: bold/italic
    # spans recurse into escape_text(), and by the time any stash() call
    # fires its value argument has already been fully evaluated (including
    # any nested escape_text() call's own complete restore), so every
    # registry entry is final the moment it exists. That makes it safe for
    # any call, inner or outer, to resolve any placeholder id it happens to
    # find in its own string -- there is no ordering dependency left.
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\x00(\d+)\x00", lambda m: _REGISTRY[int(m.group(1))], s)
    return s


def tex_escape_basic(s):
    # Order matters: backslash first is not needed since we never emit raw
    # backslashes into text passed here (stashed tokens bypass this fn).
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("$", r"\$")
    s = s.replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    s = s.replace("—", "---")
    s = s.replace("–", "--")
    # U+2212 MINUS SIGN (used for negative statistics in source tables) has
    # no glyph in base pdflatex fonts without extra unicode-math setup;
    # render as a plain hyphen-minus, which is what every negative number
    # in this manuscript's tables needs.
    s = s.replace("−", "-")
    s = s.replace("‘", "`").replace("’", "'")
    s = s.replace("“", "``").replace("”", "''")
    return s


def parse_table(lines, i):
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        rows.append(cells)
        i += 1
    if len(rows) > 1 and all(re.match(r"^:?-+:?$", c) for c in rows[1]):
        rows.pop(1)
    return rows, i


def table_to_latex(rows, caption=None):
    ncols = max(len(r) for r in rows)
    colspec = "l" * ncols
    out = ["\\begin{table}[htbp]", "\\centering", "\\small"]
    if caption:
        out.append(f"\\caption{{{escape_text(caption)}}}")
    out.append(f"\\begin{{tabular}}{{{colspec}}}")
    out.append("\\toprule")
    for r_idx, row in enumerate(rows):
        cells = [escape_text(c) for c in row] + [""] * (ncols - len(row))
        out.append(" & ".join(cells) + " \\\\")
        if r_idx == 0:
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


def convert():
    with open(SRC, encoding="utf-8") as f:
        lines = f.read().split("\n")

    # Locate major regions
    def find(pred, start=0):
        for i in range(start, len(lines)):
            if pred(lines[i]):
                return i
        return len(lines)

    abstract_start = find(lambda l: l.strip() == "# Abstract")
    intro_start = find(lambda l: l.strip().startswith("# 1. Introduction"))
    repro_start = find(lambda l: l.strip().startswith("# 8. Reproducibility Statement"))
    data_start = find(lambda l: l.strip().startswith("# 9. Data and Code Availability"))
    ethics_start = find(lambda l: l.strip().startswith("# 10. Ethics Statement"))
    refs_start = find(lambda l: l.strip() == "# References")

    # --- Title: the document's first-level heading, before the Abstract ---
    title_match = re.search(r"^#\s+(.+)$", "\n".join(lines[:abstract_start]), re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Subject-Independent Detection of ADHD from Raw EEG"

    # --- Abstract + Keywords ---
    abstract_block = lines[abstract_start + 1:intro_start]
    abstract_text_lines, keywords_line = [], ""
    for l in abstract_block:
        ls = l.strip()
        if ls.startswith("**Keywords:**"):
            keywords_line = ls.replace("**Keywords:**", "").strip()
        elif ls and ls != "---":
            abstract_text_lines.append(ls)
    abstract_text = " ".join(abstract_text_lines)
    abstract_tex = escape_text(abstract_text)
    keywords_tex = escape_text(keywords_line)

    # --- Body: Introduction through Conclusion (§1-7); §8-10 rendered
    # separately below into cas-sc backmatter sections. ---
    body_lines = lines[intro_start:repro_start]
    repro_lines = lines[repro_start + 1:data_start]
    data_lines = lines[data_start + 1:ethics_start]
    ethics_lines = lines[ethics_start + 1:refs_start]

    body_tex = render_block(body_lines)
    repro_tex = render_block(repro_lines)
    data_tex = render_block(data_lines)
    ethics_tex = render_block(ethics_lines)

    # The cas-sc backmatter moves the Reproducibility Statement out of the
    # numbered body (§8 in manuscript.md) into an appendix, so in-prose
    # references to "Section 8" would otherwise dangle; retarget them to
    # the appendix label. Markdown/DOCX keep "Section 8" verbatim, since
    # §8 remains a real numbered section in those formats.
    body_tex = body_tex.replace(
        "(Section 8, Reproducibility Statement)",
        "(Appendix~\\ref{sec:appendix:repro}, Reproducibility Appendix)",
    )
    data_tex = data_tex.replace(
        "stated in Section 8.",
        "stated in Appendix~\\ref{sec:appendix:repro}.",
    )

    preamble = build_preamble(title, abstract_tex, keywords_tex)
    backmatter = build_backmatter(repro_tex, data_tex, ethics_tex)

    full_tex = preamble + body_tex + backmatter

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(full_tex)
    print(f"Wrote {OUT} ({len(full_tex.split())} words)")


def render_block(body_lines):
    """Convert a list of markdown lines (headers, paragraphs, lists, pipe
    tables, images) into a joined LaTeX string."""
    out = []
    i = 0
    n = len(body_lines)
    while i < n:
        line = body_lines[i]
        stripped = line.strip()

        if stripped == "" or stripped == "---":
            i += 1
            continue

        # Headers: "# 1. Introduction" -> \section, "## 3.1 Dataset" -> \subsection
        m = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            # strip a leading "N." or "N.N" numbering token (LaTeX auto-numbers)
            text = re.sub(r"^\d+(\.\d+)*\.?\s+", "", text).strip()
            text_tex = escape_text(text)
            if level == 1:
                out.append(f"\\section{{{text_tex}}}")
            else:
                out.append(f"\\subsection{{{text_tex}}}")
            i += 1
            continue

        img_m = re.match(r"^!\[(.*)\]\((.*)\)$", stripped)
        if img_m:
            caption, path = img_m.group(1), img_m.group(2)
            # Manual "Figure N[a/b]. " prefixes are for the Markdown/Word
            # drafts, where nothing auto-numbers; LaTeX's own \caption
            # numbers figures automatically, so strip the prefix here to
            # avoid a doubled "Fig. 2: Figure 2. ..." caption.
            label_m = re.match(r"^Figure\s+(\d+[a-z]?)\.\s*(.*)$", caption)
            fig_label = f"fig:{label_m.group(1)}" if label_m else None
            caption_body = label_m.group(2) if label_m else caption
            out.append("\\begin{figure}[htbp]")
            out.append("\\centering")
            out.append(f"\\includegraphics[width=\\textwidth]{{{path}}}")
            out.append(f"\\caption{{{escape_text(caption_body)}}}")
            if fig_label:
                out.append(f"\\label{{{fig_label}}}")
            out.append("\\end{figure}")
            i += 1
            continue

        if stripped.startswith("|"):
            # Look back for a preceding "**Table N. ...**" caption line already emitted
            caption = None
            if out and out[-1].startswith("TABLECAPTION::"):
                caption = out.pop()[len("TABLECAPTION::"):]
            rows, next_i = parse_table(body_lines, i)
            out.append(table_to_latex(rows, caption))
            i = next_i
            continue

        if stripped.startswith("**Table") and stripped.endswith("**"):
            cap = stripped.strip("*")
            out.append(f"TABLECAPTION::{cap}")
            i += 1
            continue

        if re.match(r"^\d+\.\s", stripped):
            items = []
            while i < n and re.match(r"^\d+\.\s", body_lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s", "", body_lines[i].strip()))
                i += 1
            out.append("\\begin{enumerate}")
            for it in items:
                out.append(f"\\item {escape_text(it)}")
            out.append("\\end{enumerate}")
            continue

        if stripped.startswith("- "):
            items = []
            while i < n and body_lines[i].strip().startswith("- "):
                items.append(body_lines[i].strip()[2:])
                i += 1
            out.append("\\begin{itemize}")
            for it in items:
                out.append(f"\\item {escape_text(it)}")
            out.append("\\end{itemize}")
            continue

        # Plain paragraph: accumulate contiguous non-blank, non-structural lines
        para = [stripped]
        j = i + 1
        while j < n and body_lines[j].strip() != "" and not body_lines[j].strip().startswith(("#", "|", "---", "**Table", "![")) \
                and not re.match(r"^\d+\.\s", body_lines[j].strip()) and not body_lines[j].strip().startswith("- "):
            para.append(body_lines[j].strip())
            j += 1
        out.append(escape_text(" ".join(para)))
        out.append("")
        i = j

    return "\n\n".join(out)


def build_preamble(title, abstract_tex, keywords_tex):
    # cas-sc (Elsevier) document-class conventions, following the structure
    # of a cas-sc-class reference manuscript supplied by the user: numeric
    # natbib citations, \shorttitle/\shortauthors running heads, \title[mode=title],
    # per-author \author[n]{}/\cormark/\ead blocks, a shared \affiliation[n]{},
    # \begin{abstract}/\begin{keywords} (not elsarticle's \begin{keyword}), \maketitle.
    return r"""\documentclass[a4paper,fleqn]{cas-sc}

\usepackage[numbers,sort&compress]{natbib}
\usepackage[T1]{fontenc}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}

\begin{document}
\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}

\shorttitle{%s}
\shortauthors{Abinaya G}

\title[mode=title]{%s}

\author[1]{Abinaya G}
\cormark[1]
\ead{abinaya.g05@gmail.com}

\affiliation[1]{organization={Saveetha Engineering College},
                city={Chennai},
                state={Tamil Nadu},
                country={India}}

\cortext[cor1]{Corresponding author}

\begin{abstract}
%s
\end{abstract}

\begin{keywords}
%s
\end{keywords}

\maketitle

""" % (escape_text(title), escape_text(title), abstract_tex, keywords_tex.replace(";", " \\sep "))


def build_backmatter(repro_tex, data_tex, ethics_tex):
    # cas-sc backmatter convention: unnumbered \section*{} blocks for
    # Acknowledgements/Data Availability/Competing Interests/Funding/CRediT/
    # Ethical statement, \printcredits, then a \clearpage \appendix carrying
    # the Reproducibility Statement (kept as an appendix, as in the cas-sc
    # reference manuscript's own "Reproducibility Appendix"), and a
    # model1-num-names/references.bib bibliography.
    return r"""

\section*{Acknowledgements}
The EEG dataset analyzed in this study was collected and released by
TaghiBeyglou et al.; see the Data Availability statement below.

\section*{Data Availability}
%s

\section*{Competing Interests}
The author declares no competing interests.

\section*{Funding}
The author received no specific funding for this work.

\section*{CRediT authorship contribution statement}
\textbf{Abinaya G:} Conceptualisation, Investigation, data curation,
methodology, software, formal analysis, validation, writing -- original
draft, writing -- review \& editing.

\section*{Ethical statement}
%s

\printcredits

\clearpage
\appendix
\section{Reproducibility Appendix}
\label{sec:appendix:repro}

%s

\bibliographystyle{model1-num-names}
\bibliography{references}

\end{document}
""" % (data_tex, ethics_tex, repro_tex)


if __name__ == "__main__":
    convert()
