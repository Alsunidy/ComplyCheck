"""
ComplyCheck - SAMA Compliance Auditor
FastAPI backend, wired to the real RAG engine in engine/compliance_engine.py
(evidence pipeline -> hybrid retrieval -> OpenAI LLM judge). Requires
OPENAI_API_KEY for /run-compliance-check; mock_report.json remains only as
the seed report for testing /report and /chat before a real run.
"""
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

APP_DIR = Path(__file__).parent
MOCK_REPORT_PATH = APP_DIR / "mock_report.json"

# Ensures `import chat_handler` resolves whether this app is launched as
# `uvicorn main:app` from inside backend/ or as `uvicorn backend.main:app`
# from the project root. The engine dir is added too so compliance_engine's
# own `from common.embeddings import ...` imports resolve.
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR / "engine"))
import chat_handler  # noqa: E402
import progress  # noqa: E402
from engine import compliance_engine, evidence_pipeline  # noqa: E402

ALLOWED_POLICY_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_EVIDENCE_EXTENSIONS = evidence_pipeline.SUPPORTED_EXTENSIONS
ALLOWED_STANDARDS = {"SAMA"}

app = FastAPI(
    title="ComplyCheck - SAMA Compliance Auditor",
    description="RAG-based cybersecurity policy compliance gap analysis API",
    version="0.1.0",
)

# In-memory report store, keyed by report_id. Fine for local/dev use;
# swap for a real database when this moves beyond a single-process demo.
REPORTS_DB: dict[str, dict] = {}


class ControlResult(BaseModel):
    control_id: str
    control_domain: str
    control_text: str
    status_code: Literal["PASS", "PARTIAL", "FAIL"]
    status_label: Literal["Compliant", "Partially Compliant", "Non-Compliant"]
    matched_policy_excerpt: str
    justification: str
    recommendation: str
    # Additive fields (kept optional so the agreed base schema is unchanged).
    evidence_source: list[str] = []
    confidence_score: int | None = None


class EvidenceItem(BaseModel):
    filename: str
    type: str
    category: str = ""
    description: str = ""
    related_control: str = ""
    chunk_count: int = 0


class ComplianceReport(BaseModel):
    report_id: str
    standard: Literal["SAMA"]
    generated_at: str
    source_document: str
    # Language of the evidence and therefore of the report ("en" or "ar").
    language: Literal["en", "ar"] = "en"
    results: list[ControlResult]
    evidence_used: list[EvidenceItem] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    # Optional: without a report_id the assistant answers from the SAMA CSF
    # corpus, so users can ask questions before running any check.
    report_id: str | None = None
    question: str
    control_id: str | None = None
    # Prior turns, so follow-ups like "elaborate more" resolve to the topic
    # actually under discussion instead of being searched literally.
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    cited_control_ids: list[str]


def _load_mock_report() -> dict:
    with open(MOCK_REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Seed the store with the static mock report so GET /report/{id} has
# something to return before /run-compliance-check has ever been called.
_seed = _load_mock_report()
REPORTS_DB[_seed["report_id"]] = _seed


# The most recently processed evidence package, consumed by
# /run-compliance-check. Single-user local demo, so one slot is enough.
# {"chunks": [...provenance-tagged chunk dicts...], "manifest": [...]}
EVIDENCE_PACKAGE: dict = {}


def process_evidence_package(files: list[dict]) -> dict:
    """Run classification + extraction on a package and stash the chunks.

    The package language is detected here and drives everything downstream:
    which SAMA corpus the evidence is matched against, and the language of
    the judgements, the report and the PDF.
    """
    chunks, manifest, language = evidence_pipeline.build_evidence_chunks(
        files, compliance_engine.chunk_text
    )
    EVIDENCE_PACKAGE.clear()
    EVIDENCE_PACKAGE.update({"chunks": chunks, "manifest": manifest, "language": language})
    return {
        "status": "received",
        "files": manifest,
        "total_chunks": len(chunks),
        "language": language,
        "message": (
            f"{len(manifest)} evidence file(s) processed into {len(chunks)} "
            f"chunks (detected language: {language}), ready for compliance check."
        ),
    }


def run_compliance_engine(standard: str) -> dict:
    language = EVIDENCE_PACKAGE.get("language", "en")
    results = compliance_engine.run_evidence_compliance_check(
        EVIDENCE_PACKAGE["chunks"], language=language
    )
    manifest = EVIDENCE_PACKAGE["manifest"]
    return {
        "report_id": str(uuid.uuid4()),
        "standard": standard,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_document": ", ".join(item["filename"] for item in manifest),
        "language": language,
        "results": results,
        "evidence_used": manifest,
    }


@app.post("/upload-evidence")
async def upload_evidence(
    files: list[UploadFile] = File(...),
    metadata: str = Form("[]"),
):
    """Accept a multi-file evidence package.

    `metadata` is a JSON array of optional per-file annotations:
    [{"filename", "type", "category", "description", "related_control"}].
    Files without a metadata entry are auto-classified.
    """
    try:
        metadata_list = json.loads(metadata)
        metadata_by_name = {m.get("filename"): m for m in metadata_list if isinstance(m, dict)}
    except (json.JSONDecodeError, AttributeError):
        raise HTTPException(status_code=400, detail="metadata must be a JSON array of objects.")

    package: list[dict] = []
    for file in files:
        extension = Path(file.filename or "").suffix.lower()
        if extension not in ALLOWED_EVIDENCE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{extension}' ({file.filename}). "
                    f"Allowed types: {sorted(ALLOWED_EVIDENCE_EXTENSIONS)}"
                ),
            )
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"Uploaded file '{file.filename}' is empty.")

        meta = metadata_by_name.get(file.filename, {})
        package.append(
            {
                "filename": file.filename,
                "content": content,
                "evidence_type": meta.get("type"),
                "category": meta.get("category"),
                "description": meta.get("description"),
                "related_control": meta.get("related_control"),
            }
        )

    if not package:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    progress.start("uploading", detail=f"{len(package)}", total=len(package))
    try:
        # Off the event loop: parsing and vision OCR are blocking, and while
        # they run the server must stay free to answer GET /progress.
        result = await asyncio.to_thread(process_evidence_package, package)
        progress.finish()
    except ValueError as exc:
        progress.finish()
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:  # e.g. vision model unavailable for a screenshot
        progress.finish()
        raise HTTPException(status_code=503, detail=str(exc))
    return result


@app.post("/upload-policy")
async def upload_policy(file: UploadFile = File(...)):
    """Backward-compatible single-policy upload; wraps into an evidence package."""
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_POLICY_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed types: {sorted(ALLOWED_POLICY_EXTENSIONS)}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = await asyncio.to_thread(
            process_evidence_package,
            [{"filename": file.filename, "content": content, "evidence_type": "policy"}],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Preserve the original single-file response shape for old callers.
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
        "status": "received",
        "message": result["message"],
    }


@app.post("/run-compliance-check", response_model=ComplianceReport)
def run_compliance_check(standard: str = Form(...)):
    # Declared `def`, not `async def`: FastAPI then runs it in a worker
    # thread. As an `async def` it would block uvicorn's event loop for the
    # whole run, freezing every other request -- including the UI's progress
    # polling, and any second user.
    standard = standard.upper()
    if standard not in ALLOWED_STANDARDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid standard '{standard}'. Must be one of {sorted(ALLOWED_STANDARDS)}",
        )

    if not EVIDENCE_PACKAGE:
        raise HTTPException(
            status_code=400,
            detail="No evidence uploaded yet. Call /upload-evidence (or /upload-policy) first.",
        )

    progress.start("retrieving")
    try:
        report = run_compliance_engine(standard=standard)
        progress.finish()
    except RuntimeError as exc:
        progress.finish()
        raise HTTPException(status_code=503, detail=str(exc))
    REPORTS_DB[report["report_id"]] = report
    return report


@app.get("/report/{report_id}", response_model=ComplianceReport)
async def get_report(report_id: str):
    report = REPORTS_DB.get(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found.")
    return report


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):  # blocking LLM call -> threadpool, see above
    report = None
    if request.report_id:
        report = REPORTS_DB.get(request.report_id)
        if report is None:
            raise HTTPException(status_code=404, detail=f"Report '{request.report_id}' not found.")

    # Without a report the assistant grounds its answer in the SAMA CSF
    # controls corpus; with one, in the organization's own results.
    result = chat_handler.answer_report_question(
        question=request.question,
        report=report,
        control_id=request.control_id,
        history=[m.model_dump() for m in request.history],
    )
    return result


@app.get("/progress")
async def get_progress():
    """Live state of the current upload / compliance run, polled by the UI."""
    return progress.snapshot()


@app.get("/")
async def root():
    return {"service": "ComplyCheck - SAMA Compliance Auditor", "status": "running"}
