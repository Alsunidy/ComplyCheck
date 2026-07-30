"""
ComplyCheck - SAMA Compliance Auditor
Streamlit frontend. Talks to the FastAPI backend exclusively over HTTP --
it never reads mock_report.json or any report file directly.

The interface is bilingual (English / Arabic) via the language toggle; all
copy lives in ui_text.py. The compliance REPORT's language is separate: the
backend detects it from the uploaded evidence, so an Arabic-speaking auditor
reviewing an English policy still receives an English report.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# report_generator.py lives in ../backend relative to this file. PDF export
# runs client-side against whatever report JSON the API returned, so no new
# backend endpoint is needed for it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from report_generator import generate_compliance_pdf  # noqa: E402

from ui_text import (  # noqa: E402
    EVIDENCE_TYPES,
    EVIDENCE_TYPE_LABELS,
    LANGUAGES,
    STAGE_LABELS,
    STATUS_LABELS,
    t,
)

API_BASE_URL = os.environ.get("COMPLYCHECK_API_URL", "http://localhost:8000")

STANDARD_CODE = "SAMA"

STATUS_ROW_COLOR = {"PASS": "#d4edda", "PARTIAL": "#fff3cd", "FAIL": "#f8d7da"}
STATUS_TEXT_COLOR = {"PASS": "#155724", "PARTIAL": "#856404", "FAIL": "#721c24"}
STATUS_BADGE_COLOR = {"PASS": "green", "PARTIAL": "orange", "FAIL": "red"}

st.set_page_config(
    page_title="ComplyCheck - SAMA Compliance Auditor",
    page_icon="🛡️",
    layout="wide",
)

for key, default in [
    ("report", None),
    ("uploaded_file_info", None),
    ("chat_history", []),
    ("chat_report_id", None),
    ("ui_language", "en"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def inject_theme(rtl: bool) -> None:
    """App-wide visual polish, plus full RTL mirroring for Arabic.

    Streamlit has no RTL mode, so direction is set on the app container and on
    the widget internals that declare their own direction (uploader drop zone,
    chat input, dataframe, metrics). Latin runs -- filenames, control ids --
    are forced back to LTR so they do not read reversed inside Arabic text.
    """
    direction_css = """
        .stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2,
        .stApp h3, .stApp h4, .stApp .stMarkdown, .stApp .stAlert,
        .stApp [data-testid="stChatMessageContent"],
        .stApp [data-testid="stExpander"] summary,
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stMetricLabel"] {
            direction: rtl;
            text-align: right;
        }
        .stApp ul, .stApp ol { padding-right: 1.4rem; padding-left: 0; }
        .stApp [data-testid="stFileUploaderDropzone"] { direction: rtl; }
        .stApp [data-testid="stChatInput"] textarea {
            direction: rtl; text-align: right;
        }
        .stApp [data-testid="stChatMessage"] { flex-direction: row-reverse; }
        .stApp [data-testid="stDataFrame"] { direction: rtl; }
        .stApp code, .stApp .ltr {
            direction: ltr; unicode-bidi: embed;
            display: inline-block; text-align: left;
        }
    """ if rtl else ""

    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
        h1 {{ font-weight: 700; letter-spacing: -0.02em; margin-bottom: .1rem; }}
        h2, h3 {{ font-weight: 600; margin-top: .4rem; }}
        hr {{ margin: 1.6rem 0; opacity: .25; }}

        [data-testid="stMetric"] {{
            background: rgba(255,255,255,.035);
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 12px;
            padding: 14px 16px;
        }}
        [data-testid="stMetricValue"] {{ font-size: 1.9rem; font-weight: 700; }}
        [data-testid="stMetricLabel"] {{ opacity: .75; font-size: .85rem; }}

        [data-testid="stFileUploaderDropzone"] {{
            border: 1.5px dashed rgba(255,255,255,.18);
            border-radius: 14px;
            background: rgba(255,255,255,.02);
        }}

        .stButton > button {{
            border-radius: 10px; font-weight: 600; padding: .5rem 1.4rem;
        }}
        .stDownloadButton > button {{ border-radius: 10px; }}

        [data-testid="stExpander"] {{
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 12px;
        }}

        [data-testid="stChatMessage"] {{
            background: rgba(255,255,255,.03);
            border-radius: 14px;
            padding: .6rem .9rem;
        }}

        .cc-steps {{ line-height: 2.0; font-size: .94rem; }}
        .cc-step-done {{ opacity: .55; }}
        .cc-step-active {{ font-weight: 650; }}
        .cc-step-todo {{ opacity: .35; }}

        {direction_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


PIPELINE_STAGES = [
    "classifying", "extracting", "analyzing_images", "detecting_language",
    "indexing", "retrieving", "judging",
]


def render_progress_steps(placeholder, snapshot: dict, language: str) -> None:
    """Draw the pipeline as a checklist with the live stage highlighted."""
    current = snapshot.get("stage", "idle")
    labels = STAGE_LABELS.get(language, STAGE_LABELS["en"])
    index = (
        len(PIPELINE_STAGES) if current == "complete"
        else PIPELINE_STAGES.index(current) if current in PIPELINE_STAGES
        else -1
    )

    lines = []
    for i, stage in enumerate(PIPELINE_STAGES):
        label = labels.get(stage, stage)
        if i < index:
            lines.append(f'<div class="cc-step-done">&#10003; {label}</div>')
        elif i == index:
            extra = ""
            total = snapshot.get("total") or 0
            if total:
                extra = " &mdash; " + t(
                    language, "step_of",
                    current=min(snapshot.get("current", 0) + 1, total),
                    total=total,
                )
            detail = str(snapshot.get("detail") or "")
            if detail and not detail.isdigit():
                extra += f' <span class="ltr">{detail}</span>'
            lines.append(f'<div class="cc-step-active">&#9679; {label}{extra}</div>')
        else:
            lines.append(f'<div class="cc-step-todo">&#9675; {label}</div>')

    lines.append(
        '<div class="cc-step-todo" style="margin-top:.4rem">'
        + t(language, "elapsed", s=int(snapshot.get("elapsed", 0)))
        + "</div>"
    )
    placeholder.markdown(
        '<div class="cc-steps">' + "".join(lines) + "</div>",
        unsafe_allow_html=True,
    )


def run_with_progress(request_fn, language: str, status_label: str):
    """Run a blocking API call while polling /progress and drawing the steps.

    The request goes on a worker thread so the main thread stays free to
    refresh the step list about once a second. Failed polls are ignored --
    progress display must never break the actual work.
    """
    with st.status(status_label, expanded=True) as status:
        placeholder = st.empty()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(request_fn)
            while not future.done():
                try:
                    snap = requests.get(f"{API_BASE_URL}/progress", timeout=5).json()
                    render_progress_steps(placeholder, snap, language)
                except requests.RequestException:
                    pass
                time.sleep(0.9)
            result = future.result()  # re-raises API errors to the caller
        placeholder.empty()
        status.update(
            label=t(language, "check_complete"), state="complete", expanded=False
        )
    return result


def style_status(row: pd.Series) -> list[str]:
    color = STATUS_ROW_COLOR.get(row["_status_code"], "")
    text_color = STATUS_TEXT_COLOR.get(row["_status_code"], "")
    return [f"background-color: {color}; color: {text_color};"] * len(row)


def describe_api_error(exc: requests.RequestException) -> str:
    """Prefer the API's JSON 'detail' message over the bare HTTP error."""
    if exc.response is not None:
        try:
            return exc.response.json().get("detail", str(exc))
        except ValueError:
            pass
    return str(exc)


def render_citation_badges(cited_control_ids: list[str], results_by_id: dict) -> str:
    badges = []
    for cid in cited_control_ids:
        control = results_by_id.get(cid)
        color = STATUS_BADGE_COLOR.get(control["status_code"], "gray") if control else "gray"
        badges.append(f":{color}-background[{cid}]")
    return "  ".join(badges)


# --- Header + language toggle ---------------------------------------------
title_col, lang_col = st.columns([5, 1])
with lang_col:
    chosen = st.selectbox(
        t(st.session_state.ui_language, "language"),
        options=list(LANGUAGES.keys()),
        format_func=lambda code: LANGUAGES[code],
        index=list(LANGUAGES).index(st.session_state.ui_language),
        key="language_select",
    )
    if chosen != st.session_state.ui_language:
        st.session_state.ui_language = chosen
        st.rerun()

lang = st.session_state.ui_language
inject_theme(rtl=(lang == "ar"))

with title_col:
    st.title(t(lang, "app_title"))
    st.caption(t(lang, "app_caption"))

# --- Evidence upload -------------------------------------------------------
st.subheader(t(lang, "upload_header"))
st.markdown(t(lang, "upload_intro"))

uploaded_files = st.file_uploader(
    t(lang, "uploader_label"),
    type=[
        "pdf", "docx",              # policies / procedures / standards / reports
        "png", "jpg", "jpeg",       # screenshots
        "xlsx", "xlsm",             # compliance matrices
        "txt", "json", "yaml", "yml", "cfg", "ini", "conf", "xml", "csv", "log",
    ],
    accept_multiple_files=True,
    help=t(lang, "uploader_help"),
)
st.caption(t(lang, "standard_line"))
st.caption(t(lang, "report_language_note"))

file_metadata: list[dict] = []
if uploaded_files:
    st.markdown(t(lang, "files_in_package", n=len(uploaded_files)))
    type_options = ["auto"] + EVIDENCE_TYPES
    for i, f in enumerate(uploaded_files):
        with st.expander(f"📄 {f.name} ({f.size / 1024:.1f} KB)", expanded=False):
            if Path(f.name).suffix.lower() in {".png", ".jpg", ".jpeg"}:
                st.image(f.getvalue(), caption=f.name, width=400)
            c1, c2 = st.columns(2)
            ev_type = c1.selectbox(
                t(lang, "evidence_type"),
                options=type_options,
                format_func=lambda v: (
                    t(lang, "auto_detect") if v == "auto"
                    else EVIDENCE_TYPE_LABELS[lang][v]
                ),
                key=f"type_{i}_{f.name}",
            )
            category = c2.text_input(
                t(lang, "category"), key=f"cat_{i}_{f.name}",
                placeholder=t(lang, "category_ph"),
            )
            description = st.text_input(
                t(lang, "description"), key=f"desc_{i}_{f.name}",
                placeholder=t(lang, "description_ph"),
            )
            related_control = st.text_input(
                t(lang, "related_control"), key=f"ctrl_{i}_{f.name}",
                placeholder=t(lang, "related_control_ph"),
            )
            file_metadata.append(
                {
                    "filename": f.name,
                    "type": None if ev_type == "auto" else ev_type,
                    "category": category,
                    "description": description,
                    "related_control": related_control.strip(),
                }
            )

run_clicked = st.button(t(lang, "run_button"), type="primary", disabled=not uploaded_files)

if uploaded_files and run_clicked:
    # Both calls are wrapped in the live step display: the backend reports
    # which pipeline stage it is in, so the user sees real work (per-file
    # extraction, screenshot OCR, judging batch N of M) instead of a spinner.
    files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]

    def do_upload():
        resp = requests.post(
            f"{API_BASE_URL}/upload-evidence",
            files=files_payload,
            data={"metadata": json.dumps(file_metadata)},
            timeout=300,  # screenshots go through the vision model
        )
        resp.raise_for_status()
        return resp.json()

    def do_check():
        resp = requests.post(
            f"{API_BASE_URL}/run-compliance-check",
            data={"standard": STANDARD_CODE},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    try:
        st.session_state.uploaded_file_info = run_with_progress(
            do_upload, lang, t(lang, "processing")
        )
    except requests.RequestException as exc:
        st.error(t(lang, "upload_failed", err=describe_api_error(exc)))
        st.stop()

    try:
        st.session_state.report = run_with_progress(
            do_check, lang, t(lang, "processing")
        )
    except requests.RequestException as exc:
        st.error(t(lang, "check_failed", err=describe_api_error(exc)))
        st.stop()

    st.success(t(lang, "check_complete"))

report = st.session_state.report

# --- Results ---------------------------------------------------------------
if report:
    results = report["results"]
    df = pd.DataFrame(results)
    # Statuses are rendered from status_code so the table follows the UI
    # language even when the report itself was generated in the other one.
    status_labels = STATUS_LABELS[lang]

    st.divider()
    st.subheader(t(lang, "summary"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t(lang, "total_controls"), len(df))
    m2.metric(t(lang, "compliant"), int((df["status_code"] == "PASS").sum()))
    m3.metric(t(lang, "partial"), int((df["status_code"] == "PARTIAL").sum()))
    m4.metric(t(lang, "non_compliant"), int((df["status_code"] == "FAIL").sum()))

    report_lang = report.get("language", "en")
    st.caption(t(lang, "report_lang_badge", lang=LANGUAGES.get(report_lang, report_lang)))

    evidence_used = report.get("evidence_used", [])
    if evidence_used:
        st.markdown(t(lang, "evidence_used"))
        for item in evidence_used:
            details = [item.get("type", "")]
            if item.get("category"):
                details.append(item["category"])
            if item.get("description"):
                details.append(item["description"])
            st.markdown(f"- `{item['filename']}` — {' · '.join(d for d in details if d)}")

    st.subheader(t(lang, "results_header"))

    table_columns = {
        t(lang, "col_control"): df["control_id"],
        t(lang, "col_domain"): df["control_domain"],
        t(lang, "col_status"): df["status_code"].map(status_labels),
        t(lang, "col_gap"): df["justification"],
        t(lang, "col_recommendation"): df["recommendation"],
    }
    if "evidence_source" in df.columns:
        table_columns[t(lang, "col_evidence")] = df["evidence_source"].apply(
            lambda s: ", ".join(s) if isinstance(s, list) else (s or "")
        )
    if "confidence_score" in df.columns and df["confidence_score"].notna().any():
        table_columns[t(lang, "col_confidence")] = df["confidence_score"].apply(
            lambda c: f"{int(c)}%" if pd.notna(c) else ""
        )
    table_columns["_status_code"] = df["status_code"]

    display_df = pd.DataFrame(table_columns)
    styled = display_df.style.apply(style_status, axis=1).hide(
        axis="columns", subset=["_status_code"]
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(t(lang, "export"))
    st.download_button(
        label=t(lang, "download_pdf"),
        data=generate_compliance_pdf(report),
        file_name=f"complycheck_report_{report['report_id']}.pdf",
        mime="application/pdf",
    )
else:
    st.info(t(lang, "empty_state"))

# --- Assistant: available from the start, not only after a check ----------
# Without a report it answers from the SAMA CSF corpus; once a report exists
# it answers from the organization's own results.
st.divider()
st.subheader(t(lang, "assistant_with_report" if report else "assistant_no_report"))
st.caption(t(lang, "assistant_caption_report" if report else "assistant_caption_no_report"))

# Starting a new report clears the previous conversation.
current_report_id = report["report_id"] if report else None
if st.session_state.chat_report_id != current_report_id:
    st.session_state.chat_history = []
    st.session_state.chat_report_id = current_report_id

results_by_id = {r["control_id"]: r for r in report["results"]} if report else {}

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("cited_control_ids"):
            st.markdown(render_citation_badges(message["cited_control_ids"], results_by_id))

question = st.chat_input(t(lang, "chat_placeholder"))
if question:
    st.session_state.chat_history.append(
        {"role": "user", "content": question, "cited_control_ids": []}
    )
    # Send the prior turns (excluding the question just appended) so the
    # backend can resolve follow-ups like "elaborate more".
    payload = {
        "question": question,
        "history": [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history[:-1]
        ],
    }
    if report:
        payload["report_id"] = report["report_id"]
    try:
        with st.spinner(t(lang, "thinking")):
            chat_resp = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=120)
        chat_resp.raise_for_status()
        chat_data = chat_resp.json()
        answer = chat_data["answer"]
        cited_control_ids = chat_data.get("cited_control_ids", [])
    except requests.RequestException as exc:
        answer = t(lang, "chat_failed", err=describe_api_error(exc))
        cited_control_ids = []

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "cited_control_ids": cited_control_ids}
    )
    st.rerun()
