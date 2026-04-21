import os
import re

from fpdf import FPDF

try:
    from PIL import Image
except ImportError:
    Image = None


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_MD = os.path.join(ROOT, "report", "report.md")
REPORT_PDF = os.path.join(ROOT, "report", "report.pdf")


def clean_inline(text):
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("**", "")
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text


class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def add_paragraph(pdf, text):
    if not text.strip():
        pdf.ln(2)
        return
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, clean_inline(text))
    pdf.ln(2)


def add_heading(pdf, line):
    level = len(line) - len(line.lstrip("#"))
    title = line[level:].strip()

    if level == 1:
        pdf.set_font("Arial", "B", 18)
        pdf.ln(4)
        pdf.multi_cell(0, 9, clean_inline(title), align="C")
        pdf.ln(3)
    elif level == 2:
        pdf.set_font("Arial", "B", 14)
        pdf.ln(5)
        pdf.multi_cell(0, 7, clean_inline(title))
        pdf.ln(1)
    else:
        pdf.set_font("Arial", "B", 12)
        pdf.ln(3)
        pdf.multi_cell(0, 6, clean_inline(title))
        pdf.ln(1)


def add_table(pdf, table_lines):
    rows = []
    for line in table_lines:
        cells = [clean_inline(cell.strip()) for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(cells)

    if not rows:
        return

    n_cols = len(rows[0])
    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    if n_cols == 2:
        widths = [48, page_width - 48]
    elif n_cols == 4:
        widths = [42, 46, 46, page_width - 134]
    elif n_cols == 5:
        widths = [38, 38, 38, 38, page_width - 152]
    else:
        widths = [page_width / n_cols] * n_cols

    line_height = 7
    pdf.set_font("Arial", "B", 9)
    pdf.set_fill_color(235, 235, 235)
    for i, cell in enumerate(rows[0]):
        pdf.cell(widths[i], line_height, cell, border=1, fill=True)
    pdf.ln(line_height)

    pdf.set_font("Arial", "", 9)
    for row in rows[1:]:
        if pdf.get_y() > 270:
            pdf.add_page()
        for i, cell in enumerate(row):
            pdf.cell(widths[i], line_height, cell, border=1)
        pdf.ln(line_height)
    pdf.ln(3)


def add_image(pdf, line):
    match = re.search(r"!\[([^\]]*)\]\(([^)]+)\)", line)
    if not match:
        return

    image_path = match.group(2)
    image_path = os.path.abspath(os.path.join(os.path.dirname(REPORT_MD), image_path))
    if not os.path.exists(image_path):
        add_paragraph(pdf, f"[Missing image: {image_path}]")
        return

    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    width = page_width
    height = 85

    if Image is not None:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            height = width * img_height / img_width

    if pdf.get_y() + height > 275:
        pdf.add_page()

    pdf.image(image_path, x=pdf.l_margin, y=pdf.get_y(), w=width)
    pdf.set_y(pdf.get_y() + height + 5)


def export_pdf():
    with open(REPORT_MD, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    i = 0
    in_code = False
    table_lines = []
    paragraph = []

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            add_paragraph(pdf, " ".join(paragraph))
            paragraph = []

    while i < len(lines):
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            in_code = not in_code
            i += 1
            continue

        if in_code:
            pdf.set_font("Courier", "", 10)
            pdf.set_fill_color(245, 245, 245)
            pdf.multi_cell(0, 5, line, fill=True)
            i += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].rstrip("\n"))
                i += 1
            add_table(pdf, table_lines)
            continue

        if stripped.startswith("!["):
            flush_paragraph()
            add_image(pdf, stripped)
            i += 1
            continue

        if stripped.startswith("#"):
            flush_paragraph()
            add_heading(pdf, stripped)
            i += 1
            continue

        if stripped.startswith("- ") or re.match(r"^\d+\.\s", stripped):
            flush_paragraph()
            pdf.set_font("Arial", "", 11)
            bullet = clean_inline(stripped)
            pdf.multi_cell(0, 6, bullet)
            i += 1
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            pdf.set_font("Arial", "I", 11)
            pdf.multi_cell(0, 6, clean_inline(stripped.lstrip("> ")))
            pdf.ln(2)
            i += 1
            continue

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        paragraph.append(stripped)
        i += 1

    flush_paragraph()
    pdf.output(REPORT_PDF)
    print(REPORT_PDF)


if __name__ == "__main__":
    export_pdf()
