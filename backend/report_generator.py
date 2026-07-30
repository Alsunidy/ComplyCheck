"""
ComplyCheck - SAMA Compliance Auditor
PDF report generation (ReportLab). Kept separate from main.py so the
layout/branding can be restyled without touching API logic.
"""
from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import pdf_arabic

BRAND_NAME = "ComplyCheck"
BRAND_TAGLINE = "SAMA Compliance Auditor"
BRAND_COLOR = colors.HexColor("#1F3A5F")

STATUS_ROW_COLOR = {
    "PASS": colors.HexColor("#D4EDDA"),
    "PARTIAL": colors.HexColor("#FFF3CD"),
    "FAIL": colors.HexColor("#F8D7DA"),
}
STATUS_TEXT_COLOR = {
    "PASS": colors.HexColor("#155724"),
    "PARTIAL": colors.HexColor("#856404"),
    "FAIL": colors.HexColor("#721C24"),
}

STANDARD_DISPLAY_NAMES = {
    "SAMA": "SAMA Cyber Security Framework (CSF)",
}
STANDARD_DISPLAY_NAMES_AR = {
    "SAMA": "إطار الأمن السيبراني - البنك المركزي السعودي",
}

# All user-visible report strings, per language. Keeping them in one table
# makes the Arabic report a data change rather than a layout rewrite.
LABELS = {
    "en": {
        "tagline": BRAND_TAGLINE + " — Gap Analysis Report",
        "standard": "Standard", "source": "Source document", "report_id": "Report ID",
        "generated": "Generated at", "summary": "Summary", "evidence": "Evidence Used",
        "details": "Detailed Results", "total": "Total Controls",
        "compliant_of": "<b>{p} of {t}</b> controls fully compliant.",
        "cols": ["Control ID", "Domain", "Status", "Gap / Justification", "Recommendation"],
    },
    "ar": {
        "tagline": "مدقق الامتثال لإطار الأمن السيبراني — تقرير تحليل الفجوات",
        "standard": "الإطار", "source": "مستندات الأدلة", "report_id": "معرّف التقرير",
        "generated": "تاريخ الإصدار", "summary": "الملخص", "evidence": "الأدلة المستخدمة",
        "details": "النتائج التفصيلية", "total": "إجمالي الضوابط",
        "compliant_of": "{p} من {t} ضابطاً مطبقاً بالكامل.",
        "cols": ["رقم الضابط", "المجال", "الحالة", "الفجوة / المبرر", "التوصية"],
    },
}

STATUS_LABELS_AR = {
    "PASS": "ملتزم",
    "PARTIAL": "ملتزم جزئياً",
    "FAIL": "غير ملتزم",
}
STATUS_SHORT = {"PASS": "PASS", "PARTIAL": "PARTIAL", "FAIL": "FAIL"}
STATUS_SHORT_AR = {"PASS": "ملتزم", "PARTIAL": "جزئي", "FAIL": "غير ملتزم"}


def _text_formatter(language: str):
    """Return the text transform for a language.

    Arabic must be letter-shaped and bidi-reordered before ReportLab draws
    it; English passes through untouched.
    """
    return pdf_arabic.shape if language == "ar" else (lambda s: s)


def _build_styles(language: str = "en") -> dict:
    """Build paragraph styles; Arabic gets an Arabic-capable font and RTL text."""
    stylesheet = getSampleStyleSheet()
    is_ar = language == "ar"
    if is_ar:
        base_font = pdf_arabic.ensure_arabic_font()
        bold_font = pdf_arabic.ARABIC_FONT_BOLD_NAME
    else:
        base_font, bold_font = "Helvetica", "Helvetica-Bold"
    align = TA_RIGHT if is_ar else None

    def _a(style):
        """Apply the language font/alignment to a style."""
        style.fontName = bold_font if style.fontName.endswith("-Bold") or style.fontName == "Helvetica-Bold" else base_font
        if align is not None and style.alignment not in (TA_CENTER,):
            style.alignment = align
        return style

    styles = {
        "brand_title": ParagraphStyle(
            "BrandTitle",
            parent=stylesheet["Title"],
            textColor=BRAND_COLOR,
            fontSize=22,
            spaceAfter=2,
        ),
        "brand_subtitle": ParagraphStyle(
            "BrandSubtitle",
            parent=stylesheet["Normal"],
            fontSize=12,
            textColor=colors.HexColor("#555555"),
            spaceAfter=12,
        ),
        "section_heading": ParagraphStyle(
            "SectionHeading",
            parent=stylesheet["Heading2"],
            textColor=BRAND_COLOR,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "meta": ParagraphStyle(
            "Meta",
            parent=stylesheet["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#666666"),
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=stylesheet["Normal"],
            fontSize=8,
            leading=10,
        ),
        "cell_header": ParagraphStyle(
            "CellHeader",
            parent=stylesheet["Normal"],
            fontSize=9,
            leading=11,
            textColor=colors.white,
            fontName="Helvetica-Bold",
        ),
        "stat_number": ParagraphStyle(
            "StatNumber",
            parent=stylesheet["Normal"],
            fontSize=20,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        ),
        "stat_label": ParagraphStyle(
            "StatLabel",
            parent=stylesheet["Normal"],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
        ),
    }
    return {name: _a(style) for name, style in styles.items()}


def _build_header(report: dict, styles: dict, language: str) -> list:
    generated_at = report.get("generated_at", datetime.utcnow().isoformat())
    # An ISO timestamp inside RTL text gets visually chopped by the bidi
    # algorithm, so Arabic reports show a plain date.
    if language == "ar":
        try:
            generated_at = datetime.fromisoformat(
                generated_at.replace("Z", "+00:00")
            ).strftime("%Y/%m/%d")
        except ValueError:
            pass
    standard_code = report.get("standard", "")
    names = STANDARD_DISPLAY_NAMES_AR if language == "ar" else STANDARD_DISPLAY_NAMES
    standard_name = names.get(standard_code, standard_code)
    L = LABELS[language]
    fmt = _text_formatter(language)

    def meta_line(label: str, value: str) -> Paragraph:
        # Bold markup splits the line into separately-reordered runs in RTL,
        # which drops the space between label and value -- so Arabic uses a
        # plain line.
        text = f"{label}: {value}" if language == "ar" else f"<b>{label}:</b> {value}"
        return Paragraph(fmt(text), styles["meta"])

    return [
        Paragraph(fmt(BRAND_NAME), styles["brand_title"]),
        Paragraph(fmt(L["tagline"]), styles["brand_subtitle"]),
        meta_line(L["standard"], standard_name),
        meta_line(L["source"], report.get("source_document", "N/A")),
        meta_line(L["report_id"], report.get("report_id", "N/A")),
        meta_line(L["generated"], generated_at),
        Spacer(1, 0.5 * cm),
    ]


def _build_summary(results: list[dict], styles: dict, language: str) -> list:
    total = len(results)
    pass_count = sum(1 for r in results if r["status_code"] == "PASS")
    partial_count = sum(1 for r in results if r["status_code"] == "PARTIAL")
    fail_count = sum(1 for r in results if r["status_code"] == "FAIL")

    L = LABELS[language]
    fmt = _text_formatter(language)
    elements = [Paragraph(fmt(L["summary"]), styles["section_heading"])]
    elements.append(
        Paragraph(fmt(L["compliant_of"].format(p=pass_count, t=total)), styles["meta"])
    )
    elements.append(Spacer(1, 0.3 * cm))

    short = STATUS_SHORT_AR if language == "ar" else STATUS_SHORT
    stat_cells = [
        (str(total), L["total"], colors.HexColor("#E9ECEF"), colors.black),
        (str(pass_count), short["PASS"], STATUS_ROW_COLOR["PASS"], STATUS_TEXT_COLOR["PASS"]),
        (str(partial_count), short["PARTIAL"], STATUS_ROW_COLOR["PARTIAL"], STATUS_TEXT_COLOR["PARTIAL"]),
        (str(fail_count), short["FAIL"], STATUS_ROW_COLOR["FAIL"], STATUS_TEXT_COLOR["FAIL"]),
    ]

    number_style_cache = {}

    def stat_number_style(text_color):
        key = str(text_color)
        if key not in number_style_cache:
            number_style_cache[key] = ParagraphStyle(
                f"StatNumber_{key}",
                parent=styles["stat_number"],
                textColor=text_color,
            )
        return number_style_cache[key]

    label_row = [Paragraph(fmt(label), styles["stat_label"]) for _, label, _, _ in stat_cells]

    table_data = [
        [Paragraph(num, stat_number_style(text_color)) for num, _, _, text_color in stat_cells],
        label_row,
    ]
    col_width = 4.4 * cm
    stat_table = Table(table_data, colWidths=[col_width] * 4)
    style_commands = [
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
    ]
    for i, (_, _, bg_color, _) in enumerate(stat_cells):
        style_commands.append(("BACKGROUND", (i, 0), (i, 0), bg_color))
    stat_table.setStyle(TableStyle(style_commands))

    elements.append(stat_table)
    elements.append(Spacer(1, 0.6 * cm))
    return elements


def _build_evidence_list(evidence_used: list[dict], styles: dict, language: str) -> list:
    if not evidence_used:
        return []
    fmt = _text_formatter(language)
    elements = [Paragraph(fmt(LABELS[language]["evidence"]), styles["section_heading"])]
    for item in evidence_used:
        name = item["filename"] if language == "ar" else f"<b>{item['filename']}</b>"
        parts = [name, item.get("type", "")]
        if item.get("category"):
            parts.append(item["category"])
        if item.get("description"):
            parts.append(item["description"])
        elements.append(Paragraph(fmt(" — ".join(p for p in parts if p)), styles["meta"]))
    elements.append(Spacer(1, 0.4 * cm))
    return elements


def _build_results_table(results: list[dict], styles: dict, language: str) -> list:
    L = LABELS[language]
    fmt = _text_formatter(language)
    elements = [Paragraph(fmt(L["details"]), styles["section_heading"])]

    header = L["cols"]
    # Arabic tables read right-to-left, so the column order is mirrored.
    if language == "ar":
        header = list(reversed(header))
    table_data = [[Paragraph(fmt(h), styles["cell_header"]) for h in header]]

    for r in results:
        status_label = (
            STATUS_LABELS_AR.get(r["status_code"], r["status_label"])
            if language == "ar" else r["status_label"]
        )
        status_text = status_label
        if r.get("confidence_score") is not None:
            status_text += f" ({r['confidence_score']}%)"
        sources = r.get("evidence_source") or []
        if sources:
            status_text += "<br/><i>" + ", ".join(sources) + "</i>"
        cell_style = styles["cell"]
        if language == "ar":
            # Column widths minus the 5pt padding on each side; the long
            # prose columns are the ones that wrap onto several lines.
            def wrap(text: str, width_cm: float) -> str:
                return pdf_arabic.shape_wrapped(
                    text, cell_style.fontName, cell_style.fontSize,
                    width_cm * cm - 10,
                )
            justification = wrap(r["justification"], 6.5)
            recommendation = wrap(r["recommendation"], 6.5)
            domain = wrap(r["control_domain"], 3.2)
        else:
            justification = r["justification"]
            recommendation = r["recommendation"]
            domain = r["control_domain"]

        row = [
            Paragraph(fmt(r["control_id"]), cell_style),
            Paragraph(domain if language == "ar" else fmt(domain), cell_style),
            Paragraph(fmt(status_text), cell_style),
            Paragraph(justification if language == "ar" else fmt(justification), cell_style),
            Paragraph(recommendation if language == "ar" else fmt(recommendation), cell_style),
        ]
        table_data.append(list(reversed(row)) if language == "ar" else row)

    col_widths = [2.0 * cm, 3.2 * cm, 2.6 * cm, 6.5 * cm, 6.5 * cm]
    if language == "ar":
        col_widths = list(reversed(col_widths))
    table = Table(table_data, colWidths=col_widths, repeatRows=1)

    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    status_col = 2 if language == "en" else 2  # middle column either way
    for row_idx, r in enumerate(results, start=1):
        bg = STATUS_ROW_COLOR.get(r["status_code"])
        if bg:
            style_commands.append(("BACKGROUND", (status_col, row_idx), (status_col, row_idx), bg))

    table.setStyle(TableStyle(style_commands))
    elements.append(table)
    return elements


def generate_compliance_pdf(report: dict) -> bytes:
    """Render a ComplyCheck gap analysis report to PDF bytes.

    `report` is the dict shape returned by the /run-compliance-check and
    /report/{id} FastAPI endpoints: {report_id, standard, generated_at,
    source_document, results: [...]}.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        title=f"{BRAND_NAME} Compliance Report",
    )

    language = report.get("language", "en")
    if language not in LABELS:
        language = "en"

    styles = _build_styles(language)
    results = report.get("results", [])

    elements: list = []
    elements.extend(_build_header(report, styles, language))
    elements.append(KeepTogether(_build_summary(results, styles, language)))
    evidence_elements = _build_evidence_list(report.get("evidence_used", []), styles, language)
    if evidence_elements:
        elements.append(KeepTogether(evidence_elements))
    elements.extend(_build_results_table(results, styles, language))

    doc.build(elements)
    return buffer.getvalue()
