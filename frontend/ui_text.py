"""
frontend/ui_text.py

Every user-visible string in the Streamlit UI, in both languages.

Keeping the copy in one table (rather than inline `if lang == "ar"` branches
scattered through app.py) means adding or re-wording a label is a data edit,
and it makes it obvious when a string has been translated in one language but
forgotten in the other.

Note the distinction the UI has to communicate: this toggle changes the
INTERFACE language only. The language of the compliance report itself is
detected from the evidence the user uploads -- Arabic evidence is matched
against the Arabic SAMA corpus and produces an Arabic report, and likewise
for English -- so an Arabic-speaking auditor reviewing an English policy
still gets an English report.
"""

LANGUAGES = {"en": "English", "ar": "العربية"}

TEXT = {
    "en": {
        "app_title": "ComplyCheck",
        "app_caption": "SAMA Compliance Auditor — RAG-based evidence gap analysis against SAMA CSF",
        "language": "Language",
        "upload_header": "Upload Evidence Package",
        "upload_intro": (
            "Upload compliance evidence — multiple files are supported:\n\n"
            "✓ Policies &nbsp; ✓ Procedures &nbsp; ✓ Screenshots &nbsp; "
            "✓ Audit reports &nbsp; ✓ Configuration exports &nbsp; ✓ Compliance documents"
        ),
        "uploader_label": "Upload compliance evidence",
        "uploader_help": (
            "Drag & drop or browse. Add optional metadata per file below before "
            "running the check."
        ),
        "standard_line": "Standard: SAMA CSF",
        "report_language_note": (
            "The report language follows your evidence: Arabic evidence produces "
            "an Arabic report, English evidence an English one."
        ),
        "files_in_package": "**{n} file(s) in evidence package** — optional metadata:",
        "evidence_type": "Evidence type",
        "auto_detect": "auto-detect",
        "category": "Category (optional)",
        "category_ph": "e.g. Access Control",
        "description": "Description (optional)",
        "description_ph": "e.g. Evidence showing MFA is enabled",
        "related_control": "Related SAMA control (optional)",
        "related_control_ph": "e.g. 3.3.5",
        "run_button": "Run Compliance Check",
        "uploading": "Uploading and processing {n} evidence file(s)...",
        "running": (
            "Running the SAMA CSF compliance check... (the LLM judges all 36 "
            "controls in parallel; usually well under a minute)"
        ),
        "upload_failed": "Failed to upload evidence package: {err}",
        "check_failed": "Failed to run compliance check: {err}",
        "check_complete": "Compliance check complete.",
        "summary": "Summary",
        "total_controls": "Total controls assessed",
        "compliant": "Compliant",
        "partial": "Partially compliant",
        "non_compliant": "Non-compliant",
        "evidence_used": "**Evidence used:**",
        "results_header": "Gap Analysis Results",
        "col_control": "Control ID",
        "col_domain": "Domain",
        "col_status": "Status",
        "col_gap": "Gap / Justification",
        "col_recommendation": "Recommendation",
        "col_evidence": "Evidence",
        "col_confidence": "Confidence",
        "export": "Export",
        "download_pdf": "Download report as PDF",
        "empty_state": (
            "Upload your compliance evidence package and run a compliance check "
            "to see results."
        ),
        "assistant_no_report": "Ask the SAMA CSF assistant",
        "assistant_with_report": "Ask about your results",
        "assistant_caption_no_report": (
            'Ask anything about the SAMA Cyber Security Framework — e.g. '
            '"what does SAMA require for incident management?"'
        ),
        "assistant_caption_report": (
            'Ask about your assessment — e.g. "which controls need urgent attention?"'
        ),
        "chat_placeholder": "Ask about SAMA CSF or your results...",
        "thinking": "Thinking...",
        "chat_failed": "Failed to get an answer: {err}",
        "report_lang_badge": "Report language: {lang}",
        "processing": "Processing your evidence",
        "step_of": "{current} of {total}",
        "elapsed": "{s}s elapsed",
    },
    "ar": {
        "app_title": "ComplyCheck",
        "app_caption": "مدقق الامتثال لإطار الأمن السيبراني — تحليل فجوات الأدلة وفق إطار البنك المركزي السعودي",
        "language": "اللغة",
        "upload_header": "رفع حزمة الأدلة",
        "upload_intro": (
            "ارفع أدلة الامتثال — يمكن رفع عدة ملفات:\n\n"
            "✓ السياسات &nbsp; ✓ الإجراءات &nbsp; ✓ لقطات الشاشة &nbsp; "
            "✓ تقارير التدقيق &nbsp; ✓ ملفات الإعدادات &nbsp; ✓ مستندات الامتثال"
        ),
        "uploader_label": "رفع أدلة الامتثال",
        "uploader_help": (
            "اسحب الملفات وأفلتها أو تصفّح. يمكنك إضافة بيانات وصفية اختيارية "
            "لكل ملف قبل تشغيل الفحص."
        ),
        "standard_line": "الإطار: إطار الأمن السيبراني (ساما)",
        "report_language_note": (
            "لغة التقرير تتبع أدلتك: الأدلة العربية تُنتج تقريراً عربياً، "
            "والأدلة الإنجليزية تُنتج تقريراً إنجليزياً."
        ),
        "files_in_package": "**{n} ملف في حزمة الأدلة** — بيانات وصفية اختيارية:",
        "evidence_type": "نوع الدليل",
        "auto_detect": "كشف تلقائي",
        "category": "التصنيف (اختياري)",
        "category_ph": "مثال: إدارة الوصول",
        "description": "الوصف (اختياري)",
        "description_ph": "مثال: دليل على تفعيل المصادقة متعددة العوامل",
        "related_control": "الضابط المرتبط (اختياري)",
        "related_control_ph": "مثال: 3.3.5",
        "run_button": "تشغيل فحص الامتثال",
        "uploading": "جارٍ رفع ومعالجة {n} ملف من الأدلة...",
        "running": (
            "جارٍ تنفيذ فحص الامتثال لإطار ساما... (يقيّم النموذج جميع الضوابط "
            "الـ36 بالتوازي، وعادةً يستغرق أقل من دقيقة)"
        ),
        "upload_failed": "تعذّر رفع حزمة الأدلة: {err}",
        "check_failed": "تعذّر تنفيذ فحص الامتثال: {err}",
        "check_complete": "اكتمل فحص الامتثال.",
        "summary": "الملخص",
        "total_controls": "إجمالي الضوابط المقيّمة",
        "compliant": "ملتزم",
        "partial": "ملتزم جزئياً",
        "non_compliant": "غير ملتزم",
        "evidence_used": "**الأدلة المستخدمة:**",
        "results_header": "نتائج تحليل الفجوات",
        "col_control": "رقم الضابط",
        "col_domain": "المجال",
        "col_status": "الحالة",
        "col_gap": "الفجوة / المبرر",
        "col_recommendation": "التوصية",
        "col_evidence": "الأدلة",
        "col_confidence": "الثقة",
        "export": "التصدير",
        "download_pdf": "تحميل التقرير بصيغة PDF",
        "empty_state": "ارفع حزمة أدلة الامتثال وشغّل الفحص لعرض النتائج.",
        "assistant_no_report": "اسأل مساعد إطار ساما",
        "assistant_with_report": "اسأل عن نتائجك",
        "assistant_caption_no_report": (
            "اسأل عن أي شيء في إطار الأمن السيبراني — مثال: "
            "«ما متطلبات ساما لإدارة الحوادث؟»"
        ),
        "assistant_caption_report": (
            "اسأل عن تقييمك — مثال: «ما الضوابط التي تحتاج معالجة عاجلة؟»"
        ),
        "chat_placeholder": "اسأل عن إطار ساما أو عن نتائجك...",
        "thinking": "جارٍ التفكير...",
        "chat_failed": "تعذّر الحصول على إجابة: {err}",
        "report_lang_badge": "لغة التقرير: {lang}",
        "processing": "جارٍ معالجة أدلتك",
        "step_of": "{current} من {total}",
        "elapsed": "مضى {s} ثانية",
    },
}

# Live pipeline stages shown while a check runs. Keys must match the stage
# names the backend reports through GET /progress (see backend/progress.py).
STAGE_LABELS = {
    "en": {
        "uploading": "Receiving evidence files",
        "classifying": "Classifying evidence types",
        "extracting": "Extracting text from documents",
        "analyzing_images": "Reading screenshots (OCR + vision)",
        "detecting_language": "Detecting evidence language",
        "indexing": "Building the hybrid search index",
        "retrieving": "Matching evidence to SAMA controls",
        "judging": "Auditing controls with the LLM",
        "complete": "Done",
        "idle": "Waiting",
    },
    "ar": {
        "uploading": "استلام ملفات الأدلة",
        "classifying": "تصنيف أنواع الأدلة",
        "extracting": "استخراج النصوص من المستندات",
        "analyzing_images": "قراءة لقطات الشاشة (تحليل بصري)",
        "detecting_language": "تحديد لغة الأدلة",
        "indexing": "بناء فهرس البحث الهجين",
        "retrieving": "مطابقة الأدلة بضوابط ساما",
        "judging": "تدقيق الضوابط بالذكاء الاصطناعي",
        "complete": "اكتمل",
        "idle": "في الانتظار",
    },
}

# Status labels shown in the results table, keyed by the API's status_code.
STATUS_LABELS = {
    "en": {"PASS": "Compliant", "PARTIAL": "Partially Compliant", "FAIL": "Non-Compliant"},
    "ar": {"PASS": "ملتزم", "PARTIAL": "ملتزم جزئياً", "FAIL": "غير ملتزم"},
}

# Evidence types offered in the per-file metadata editor. The value sent to
# the API is always the English key; only the display label is translated.
# Must stay in sync with EVIDENCE_TYPES in backend/engine/evidence_pipeline.py
# (duplicated rather than imported so the UI has no backend import at all).
EVIDENCE_TYPES = ["policy", "procedure", "standard", "screenshot",
                  "audit_report", "compliance_matrix", "configuration", "other"]

EVIDENCE_TYPE_LABELS = {
    "en": {t: t.replace("_", " ") for t in EVIDENCE_TYPES},
    "ar": {
        "policy": "سياسة",
        "procedure": "إجراء",
        "standard": "معيار",
        "screenshot": "لقطة شاشة",
        "audit_report": "تقرير تدقيق",
        "compliance_matrix": "مصفوفة امتثال",
        "configuration": "ملف إعدادات",
        "other": "أخرى",
    },
}


def t(language: str, key: str, **kwargs) -> str:
    """Look up a UI string, falling back to English if a key is untranslated."""
    value = TEXT.get(language, TEXT["en"]).get(key) or TEXT["en"][key]
    return value.format(**kwargs) if kwargs else value
