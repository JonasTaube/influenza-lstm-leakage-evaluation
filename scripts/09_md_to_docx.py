# -*- coding: utf-8 -*-
"""Convert manuscript_draft.md to a submission-ready .docx (E&I style)."""
import re
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SRC = r"../manuscript_draft.md"
DST = r"../manuscript_draft.docx"

BODY_FONT = "Times New Roman"
CJK_FONT = "宋体"


def set_run_font(run, size=12, bold=None, italic=None):
    run.font.name = BODY_FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), CJK_FONT)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


INLINE_RE = re.compile(r"(\*\*.+?\*\*|\*[^*\n]+?\*)")


def add_runs(par, text, size=12, base_bold=False, base_italic=False):
    """Parse **bold** and *italic* inline markers into runs."""
    for tok in INLINE_RE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            r = par.add_run(tok[2:-2])
            set_run_font(r, size, bold=True, italic=base_italic)
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = par.add_run(tok[1:-1])
            set_run_font(r, size, bold=base_bold, italic=True)
        else:
            r = par.add_run(tok)
            set_run_font(r, size, bold=base_bold, italic=base_italic)


def body_par(doc, text, size=12, bold=False, italic=False, align=None,
             space_after=6, line=1.5, indent=None, first_line=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    if line:
        pf.line_spacing = line
    if align is not None:
        p.alignment = align
    if indent is not None:
        pf.left_indent = Cm(indent)
    if first_line is not None:
        pf.first_line_indent = Cm(first_line)
    add_runs(p, text, size=size, base_bold=bold, base_italic=italic)
    return p


def set_cell_borders(cell, top=False, bottom=False):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge, on in (("top", top), ("bottom", bottom)):
        el = OxmlElement(f"w:{edge}")
        if on:
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "6")
            el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "nil")
        borders.append(el)
    for edge in ("left", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        borders.append(el)
    tcPr.append(borders)


def add_md_table(doc, rows, size=9.5, numeric_right=True):
    """rows: list of lists of cell strings; row 0 = header."""
    ncol = len(rows[0])
    table = doc.add_table(rows=len(rows), cols=ncol)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for i, row in enumerate(rows):
        for j, cell_text in enumerate(row):
            cell = table.cell(i, j)
            cell.paragraphs[0].text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            is_header = i == 0
            txt = cell_text.strip()
            if numeric_right and j > 0 and not is_header:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            add_runs(p, txt, size=size, base_bold=is_header)
            set_cell_borders(cell, top=(i == 0), bottom=(i == 0 or i == len(rows) - 1))
    return table


def main():
    with open(SRC, encoding="utf-8") as f:
        lines = f.read().split("\n")

    doc = Document()
    # page setup: A4, 2.5 cm margins
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.5)
    sec.left_margin = sec.right_margin = Cm(2.5)

    # default style
    style = doc.styles["Normal"]
    style.font.name = BODY_FONT
    style.element.rPr.rFonts.set(qn("w:eastAsia"), CJK_FONT)
    style.font.size = Pt(12)

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()

        if not line.strip():
            i += 1
            continue

        # skip horizontal rules
        if re.fullmatch(r"-{3,}", line.strip()):
            i += 1
            continue

        # markdown table block
        if line.lstrip().startswith("|"):
            tbl_lines = []
            while i < n and lines[i].lstrip().startswith("|"):
                tbl_lines.append(lines[i].strip())
                i += 1
            rows = []
            for tl in tbl_lines:
                cells = [c.strip() for c in tl.strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    continue  # separator row
                rows.append(cells)
            if rows:
                add_md_table(doc, rows)
                body_par(doc, "", space_after=6, line=1.0)
            continue

        # H1
        if line.startswith("# "):
            txt = line[2:].strip()
            body_par(doc, txt, size=13, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
                     space_after=12, line=1.15)
            i += 1
            continue

        # H2
        if line.startswith("## "):
            txt = line[3:].strip()
            body_par(doc, "", space_after=0, line=1.0)
            p = body_par(doc, txt, size=12.5, bold=True, space_after=8, line=1.15)
            p.paragraph_format.space_before = Pt(14)
            i += 1
            continue

        # H3
        if line.startswith("### "):
            txt = line[4:].strip()
            body_par(doc, "", space_after=0, line=1.0)
            p = body_par(doc, txt, size=12, bold=True, italic=True, space_after=6, line=1.15)
            p.paragraph_format.space_before = Pt(10)
            i += 1
            continue

        # numbered list item (e.g. update schedules, affiliations, references)
        m = re.match(r"^(\d+)\.\s+(.*)$", line.strip())
        if m:
            txt = m.group(2)
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.space_after = Pt(6)
            pf.line_spacing = 1.5
            pf.left_indent = Cm(1.0)
            pf.first_line_indent = Cm(-0.6)
            r = p.add_run(m.group(1) + ". ")
            set_run_font(r, 12)
            add_runs(p, txt, size=12)
            i += 1
            continue

        # bullet list
        if line.strip().startswith("- "):
            txt = line.strip()[2:]
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.space_after = Pt(6)
            pf.line_spacing = 1.5
            pf.left_indent = Cm(1.0)
            pf.first_line_indent = Cm(-0.5)
            r = p.add_run("• ")
            set_run_font(r, 12)
            add_runs(p, txt, size=12)
            i += 1
            continue

        # equation line (indented, starts with spaces)
        if line.startswith("  ŷ") or line.strip().startswith("ŷ_"):
            p = body_par(doc, line.strip(), size=12, italic=False,
                         align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8, line=1.15)
            i += 1
            continue

        # normal paragraph
        body_par(doc, line.strip(), size=12, space_after=6, line=1.5,
                 first_line=0.75)
        i += 1

    # hanging indent + tighter spacing for the References section paragraphs
    in_refs = False
    for p in doc.paragraphs:
        if p.text.strip().startswith("References"):
            in_refs = True
            continue
        if in_refs and re.match(r"^\d+\.\s", p.text.strip()):
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.left_indent = Cm(1.0)
            p.paragraph_format.first_line_indent = Cm(-1.0)

    doc.save(DST)
    print("saved:", DST)


if __name__ == "__main__":
    main()
