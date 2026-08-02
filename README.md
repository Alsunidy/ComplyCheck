# ComplyCheck - SAMA Compliance Auditor

RAG-based compliance tool that audits a company's **evidence package**
(policies, procedures, standards, screenshots, audit reports, Excel
compliance matrices, configuration exports) against the SAMA CSF
cybersecurity control framework and produces a gap analysis report — the way
a GRC team would run a real SAMA CSF audit, not just a PDF analyzer.

The FastAPI backend is wired to the **real RAG engine** in
[`backend/engine/`](backend/engine) (integrated from the teammate's
`compliance_rag_1` project). The evidence pipeline stages:

```
Multiple Evidence Files
  -> Document Classification   (user-declared type or auto-detected)
  -> Text Extraction / OCR     (PDF/DOCX/XLSX/configs parsed; screenshots
                                OCR'd + described by an OpenAI vision model)
  -> Evidence Understanding    (screenshots: what the image demonstrates)
  -> SAMA CSF Control Mapping  (multilingual embeddings, source-aware
                                retrieval so every evidence file is considered)
  -> Compliance Assessment     (OpenAI LLM judge, several controls per
                                request: does the evidence PROVE the control
                                is implemented? PASS/PARTIAL/FAIL +
                                confidence score + cited evidence sources)
```

`backend/mock_report.json` remains only as a seed report for testing
`/report` and `/chat` before a real run.

## Project structure

```
backend/
  main.py                  FastAPI app (endpoints, wired to the engine)
  chat_handler.py          LLM chat: SAMA CSF Q&A + questions about your report
  report_generator.py      ReportLab PDF report builder (EN + AR)
  pdf_arabic.py            Arabic font, letter shaping and RTL for the PDF
  progress.py              Live pipeline progress, served at GET /progress
  mock_report.json         Seed report (testing /report and /chat)
  engine/                  RAG engine (built on the teammate's foundation)
    compliance_engine.py     control mapping + OpenAI LLM judge
    evidence_pipeline.py     classification, extraction, vision OCR
    retrieval.py             hybrid search (Chroma vectors + BM25, RRF)
    llm_client.py            OpenAI client with model fallback chain
    arabic_text.py           language detection + Arabic ligature repair
    common/embeddings.py     multilingual embedding model (fastembed)
    ingestion/               SAMA CSF controls (EN + AR jsonl) + parsers
    data/sama_csf/           SAMA CSF source documents
frontend/
  app.py                   Streamlit UI
  ui_text.py               every UI string, in English and Arabic
tests/                     pytest suite (see Tests below)
sample_evidence/           sample files to try the tool with (see below)
archive/                   kept for reference, not used at runtime
requirements.txt
requirements-dev.txt        test dependencies
.env.example               copy to .env and add your OpenAI key
```

### sample_evidence/

Everything needed to demo the tool, in one folder:

| File | Type | Language |
|---|---|---|
| `sample_policy.pdf` | policy | English |
| `security_policy.docx` | policy | English |
| `Cybersecurity in Project Management.docx` | procedure | English |
| `Vulnerability and Patch Management.docx` | procedure | English |
| `سياسة أمن المعلومات لشركة Acme.docx` | policy | Arabic |
| `سياسة_أمن_المعلومات_عربي.docx` | policy | Arabic |
| `mfa_screenshot.png` | screenshot (OCR + vision) | English |
| `compliance_matrix.xlsx` | compliance matrix | English |
| `password_policy_config.yaml` | configuration export | English |

Upload the English set for an English report, or an Arabic policy for a
fully Arabic report — the language is detected from the evidence.

### archive/

Not imported by the app; kept only so the decisions behind it stay
reviewable.

| File | Why it's here |
|---|---|
| `controls_ar.jsonl.bak` | Arabic corpus before the ligature repairs |
| `build_index.py` | static Chroma index, superseded by in-memory `retrieval.py` |
| `requirements_engine_orig.txt` | the engine's original dependency pins |

## Prerequisites

- Python 3.10+
- pip
- An OpenAI API key with available credit — https://platform.openai.com/api-keys
- Internet access on first run: the embedding model (~220MB) is downloaded
  from Hugging Face automatically and cached locally.

No other system-level dependencies are required — everything installs with a
plain `pip install`.

## Environment variables

| Variable                | Default                 | Used by  | Purpose                                  |
|--------------------------|--------------------------|----------|-------------------------------------------|
| `OPENAI_API_KEY`        | (required)              | backend  | LLM judge + screenshot analysis. Put it in a `.env` file at the project root: `OPENAI_API_KEY=sk-...` |
| `COMPLYCHECK_JUDGE_MODEL` | (auto fallback chain) | backend | Preferred judge model; on rate limits/unavailability the engine falls back through `gpt-4.1-mini` → `gpt-4o-mini` → `gpt-4.1` |
| `COMPLYCHECK_VISION_MODEL` | (auto fallback chain) | backend | Preferred vision model for screenshots; same fallback chain |
| `COMPLYCHECK_JUDGE_BATCH_SIZE` | `6`              | backend  | Controls judged per LLM request |
| `COMPLYCHECK_JUDGE_CONCURRENCY` | `6`             | backend  | Batches sent in parallel |
| `COMPLYCHECK_API_URL`   | `http://localhost:8000` | frontend | Base URL the Streamlit app calls for the API |

## Install steps

From the project root, on a clean machine:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Running the backend (FastAPI)

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The API is now available at `http://localhost:8000` (interactive docs at
`http://localhost:8000/docs`).

## Running the frontend (Streamlit)

In a second terminal (with the same virtual environment activated):

```bash
cd frontend
streamlit run app.py
```

This opens the UI at `http://localhost:8501`. If the FastAPI backend runs on
a different host/port, set `COMPLYCHECK_API_URL` before launching:

```bash
# macOS/Linux
export COMPLYCHECK_API_URL="http://localhost:8000"
streamlit run app.py

# Windows (PowerShell)
$env:COMPLYCHECK_API_URL = "http://localhost:8000"
streamlit run app.py
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

177 tests, ~4 seconds, **no API key and no network required** — every OpenAI
call is stubbed and the embedding model is replaced with a deterministic
fake, so a grader can run the suite immediately after `pip install`.

| File | Covers |
|---|---|
| `test_arabic_text.py` | ligature repair, language detection, and that the shipped Arabic corpus is free of extraction defects |
| `test_retrieval.py` | hybrid search: tokenizing, ranking, and the RRF fusion rule |
| `test_evidence_pipeline.py` | evidence classification, per-row spreadsheet chunking, provenance tagging, package language |
| `test_compliance_engine.py` | control loading, verdict parsing/clamping, batching, the no-evidence path |
| `test_llm_client.py` | model fallback chain, retry-delay parsing, fail-fast on exhausted quota |
| `test_chat_handler.py` | grounding, citation filtering, language routing, follow-up query rewriting |
| `test_api.py` | every endpoint: validation, error codes, backward compatibility |
| `test_report_and_ui.py` | PDF generation (EN + AR), Arabic shaping, progress state, UI copy completeness |

Several tests are regression guards for defects found during development,
and are commented as such — for example that RRF must ignore zero-scoring
BM25 results (which once made "multi-factor authentication" retrieve the
hiring-checks sentence), and that the Arabic repair must leave "إلى" and
"إلكتروني" untouched.

`tests/TEST_RESULTS.txt` holds the recorded output of a full run.

### Integration tests (optional)

Four tests exercise the real OpenAI API and are excluded from the default
run. With `OPENAI_API_KEY` set:

```bash
pytest -m integration
```

They cost a few cents and take ~35s: a full 36-control audit of
`sample_evidence/sample_policy.pdf`, an English and an Arabic chat answer,
and a grounding check that the assistant refuses a question whose answer is
not in the retrieved context.

## Sample request/response

**Upload an evidence package (required before running a check):**

Multiple files of mixed types, with optional per-file metadata (JSON array
matched to files by `filename`; every field is optional):

```bash
curl -X POST "http://localhost:8000/upload-evidence" \
  -F "files=@sample_evidence/sample_policy.pdf;type=application/pdf" \
  -F "files=@sample_evidence/mfa_screenshot.png;type=image/png" \
  -F "files=@sample_evidence/compliance_matrix.xlsx" \
  -F 'metadata=[{"filename":"mfa_screenshot.png","type":"screenshot","category":"Access Control","description":"Evidence showing MFA is enabled","related_control":"3.3.5"}]'
```

```json
{
  "status": "received",
  "files": [
    {"filename": "sample_policy.pdf", "type": "policy", "category": "", "description": "", "related_control": "", "chunk_count": 6},
    {"filename": "mfa_screenshot.png", "type": "screenshot", "category": "Access Control", "description": "Evidence showing MFA is enabled", "related_control": "3.3.5", "chunk_count": 1},
    {"filename": "compliance_matrix.xlsx", "type": "compliance_matrix", "category": "", "description": "", "related_control": "", "chunk_count": 1}
  ],
  "total_chunks": 8,
  "message": "3 evidence file(s) processed into 8 chunks, ready for compliance check."
}
```

Supported evidence types: PDF, DOCX, PNG/JPG screenshots (OCR'd and
described by the vision model), XLSX/XLSM matrices, and text/config files
(txt, json, yaml, cfg, ini, conf, xml, csv, log).

**Upload a single policy document (backward-compatible endpoint):**

```bash
curl -X POST "http://localhost:8000/upload-policy" \
  -F "file=@sample_evidence/sample_policy.pdf;type=application/pdf"
```

```json
{"filename": "sample_policy.pdf", "content_type": "application/pdf", "size_bytes": 3063, "status": "received", "message": "1 evidence file(s) processed into 6 chunks, ready for compliance check."}
```

**Run a compliance check** (grades all 36 SAMA CSF controls with the LLM
judge — takes ~2 minutes):

```bash
curl -X POST "http://localhost:8000/run-compliance-check" \
  -F "standard=SAMA"
```

Response (truncated):

```json
{
  "report_id": "b3e1a4d2-...",
  "standard": "SAMA",
  "generated_at": "2026-07-23T09:12:00.123456+00:00",
  "source_document": "sample_policy.pdf, mfa_screenshot.png, compliance_matrix.xlsx",
  "results": [
    {
      "control_id": "3.1.1",
      "control_domain": "Cyber Security Governance",
      "control_text": "The organization shall establish, approve and maintain a cyber security policy...",
      "status_code": "PASS",
      "status_label": "Compliant",
      "matched_policy_excerpt": "The Information Security Policy is approved annually by the Board Risk Committee...",
      "justification": "The uploaded policy explicitly documents board-level approval...",
      "recommendation": "No action required. Continue annual review cadence...",
      "evidence_source": ["sample_policy.pdf", "compliance_matrix.xlsx"],
      "confidence_score": 85
    }
  ],
  "evidence_used": [
    {"filename": "sample_policy.pdf", "type": "policy", "category": "", "description": "", "related_control": "", "chunk_count": 6}
  ]
}
```

`evidence_source` and `confidence_score` are additive fields — the base
schema agreed for the team's report format is unchanged.

**Fetch a saved report by ID:**

```bash
curl "http://localhost:8000/report/b3e1a4d2-..."
```

**Ask a question about a report (`/chat`):**

`answer_report_question()` in [`backend/chat_handler.py`](backend/chat_handler.py)
is currently a rule-based placeholder standing in for the real LLM call
(same LLM setup as `compliance_engine.py`). It always answers strictly from
the report's own data and returns `"I don't have that information in this
report."` when nothing relevant is found, so the endpoint's behavior won't
change once the real LLM is wired in.

General question, no `control_id` (uses the whole report as context):

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"report_id": "b3e1a4d2-...", "question": "Which controls need urgent attention?"}'
```

```json
{
  "answer": "The following controls need urgent attention (non-compliant):\n- 3.2.4 (Cyber Security Strategy): The policy does not define any KPIs, metrics, or a reporting cadence for tracking strategy execution, and no related appendix or annex was found.\n- 3.3.15 (Human Resources Security): ...\n- 3.5.9 (Third Party Cyber Security): ...",
  "cited_control_ids": ["3.2.4", "3.3.15", "3.5.9"]
}
```

Scoped question with `control_id` (context limited to that control only):

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"report_id": "b3e1a4d2-...", "question": "Why did this fail?", "control_id": "3.2.4"}'
```

```json
{
  "answer": "For control 3.2.4 (Cyber Security Strategy): The policy does not define any KPIs, metrics, or a reporting cadence for tracking strategy execution, and no related appendix or annex was found.",
  "cited_control_ids": ["3.2.4"]
}
```
