"""
engine/compliance_engine.py

Control mapping and compliance assessment. Builds on:
  - common/embeddings.py        (multilingual fastembed model, EN+AR)
  - ingestion/controls*.jsonl   (36 parsed SAMA CSF controls, English + Arabic)
  - retrieval.py                (hybrid Chroma vector + BM25 keyword search)

Pipeline per run:
  1. Evidence chunks (from evidence_pipeline) are indexed in a hybrid
     retriever; for each SAMA control the most relevant chunks are retrieved
     by fusing semantic and keyword rankings (RRF).
  2. Controls are judged PASS / PARTIAL / FAIL by the OpenAI LLM judge in
     BATCHES (several controls per request): batching turns 36 small calls
     into ~6 larger ones, which cuts both wall-clock time and per-request
     overhead, and keeps the run well inside per-minute rate limits.

Requires OPENAI_API_KEY in the environment (or a .env file at the project
root) -- create one at https://platform.openai.com/api-keys.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

import progress
from retrieval import HybridRetriever

ENGINE_DIR = Path(__file__).resolve().parent
CONTROLS_PATH = ENGINE_DIR / "ingestion" / "controls.jsonl"

load_dotenv(ENGINE_DIR.parent.parent / ".env")

# Optional preferred judge model; llm_client walks its fallback chain from
# here when quotas run out or a model is retired.
JUDGE_MODEL = os.environ.get("COMPLYCHECK_JUDGE_MODEL")
# Evidence for one control is often spread across several sections of a
# policy (e.g. committee composition in one section, CISO requirements in
# another). Too small a k makes the judge deny content that exists but was
# never shown to it -- with batched calls the extra excerpts are cheap.
TOP_K_CHUNKS = 6
# Below this best-similarity score (across ALL evidence) we treat the control
# as having no relevant coverage at all and skip the LLM call (automatic FAIL).
NO_MATCH_THRESHOLD = 0.25
# Higher bar used only for the "always give every evidence source a chance"
# fairness pass below. Short, terse evidence (spreadsheet rows, config lines)
# sits at a much higher baseline similarity to almost every control than
# verbose policy prose does with this embedding model, so reusing
# NO_MATCH_THRESHOLD there causes over-matching (e.g. every row of a
# compliance matrix "matching" every control). For evidence you want
# guaranteed to be considered for a specific control regardless of embedding
# similarity, tag it with that control's related_control instead.
SOURCE_FAIRNESS_THRESHOLD = 0.45
# Several controls are judged per request: 6 controls/batch keeps each
# request a comfortable size while turning 36 judgements into ~6 requests,
# which is both faster and cheaper than one call per control.
JUDGE_BATCH_SIZE = int(os.environ.get("COMPLYCHECK_JUDGE_BATCH_SIZE", "6"))
# Batches are independent, so they're sent concurrently.
JUDGE_CONCURRENCY = int(os.environ.get("COMPLYCHECK_JUDGE_CONCURRENCY", "6"))

STATUS_LABELS = {
    "PASS": "Compliant",
    "PARTIAL": "Partially Compliant",
    "FAIL": "Non-Compliant",
}

JUDGE_SYSTEM_PROMPT = (
    "You are a SAMA Cyber Security Framework compliance auditor reviewing an "
    "evidence package submitted by a company (policies, procedures, standards, "
    "screenshots, audit reports, compliance matrices, configuration exports). "
    "You are given SEVERAL SAMA CSF controls; for each one, the most relevant "
    "evidence excerpts are provided, labelled with their source file and "
    "evidence type.\n\n"
    "Do not merely match keywords. Reason like an auditor: does this evidence "
    "PROVE that the organization actually implemented each SAMA CSF "
    "requirement? A policy stating an intention is weaker proof than a "
    "screenshot or audit report demonstrating the control operating in "
    "practice. Judge each control INDEPENDENTLY, using ONLY the excerpts "
    "provided for it.\n\n"
    "Respond with a single JSON object mapping each control_id to an object "
    "with exactly these keys:\n"
    '  "status_code": "PASS" | "PARTIAL" | "FAIL"\n'
    '  "justification": 1-3 sentences explaining the judgement, grounded in the evidence\n'
    '  "recommendation": 1-3 sentences; concrete remediation advice, or "No action required." if PASS\n'
    '  "confidence_score": integer 0-100; how confident you are in this judgement given the '
    "strength, specificity and type of the evidence\n\n"
    'Example response shape: {"3.1.1": {"status_code": "PASS", ...}, "3.1.2": {...}}\n\n'
    "Grading guide: PASS = the evidence convincingly demonstrates the "
    "control's core requirements are implemented; PARTIAL = some requirements "
    "are evidenced but with gaps or only stated intent without proof of "
    "implementation; FAIL = the evidence does not meaningfully address the "
    "control. Do not invent evidence content that is not in the excerpts.\n\n"
    "CRITICAL accuracy rule: before claiming any requirement is missing or "
    "not addressed, re-scan EVERY excerpt for that control -- if a statement "
    "covering it appears anywhere in the excerpts, you MUST acknowledge it "
    "as present. Falsely reporting present evidence as missing is the worst "
    "error you can make. Only cite a gap for requirements genuinely absent "
    "from all excerpts, and remember the excerpts are a retrieved subset, so "
    'phrase absences as "not found in the provided evidence" rather than '
    "asserting the organization lacks it entirely.\n\n"
    "Every control_id in the input MUST appear in the response."
)

# Appended to the judge prompt for Arabic evidence packages so the report the
# user reads back is in the language they submitted.
ARABIC_OUTPUT_INSTRUCTION = (
    "\n\nIMPORTANT: The evidence and controls are in Arabic. Write the "
    '"justification" and "recommendation" values in formal Modern Standard '
    "Arabic. Keep the JSON keys, the control_id values and the status_code "
    "values in English exactly as specified."
)


def chunk_text(text: str) -> list[str]:
    """Split extracted text into retrieval-sized chunks.

    PDF extractors often emit no blank lines between paragraphs, so split on
    individual lines and re-accumulate into chunks of a few hundred chars,
    starting a new chunk at numbered/heading-like lines when possible.
    """
    heading_re = re.compile(r"^(\d+[\.\)]|[A-Z][A-Za-z ]{2,40}:?$)")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    chunks: list[str] = []
    buffer = ""
    for line in lines:
        starts_section = bool(heading_re.match(line))
        if buffer and (len(buffer) >= 600 or (starts_section and len(buffer) >= 200)):
            chunks.append(buffer)
            buffer = ""
        buffer = f"{buffer}\n{line}".strip() if buffer else line
    if buffer:
        chunks.append(buffer)
    return chunks


_CONTROLS_CACHE: dict[str, list[dict]] = {}


def _load_controls(language: str = "en") -> list[dict]:
    """Load the SAMA CSF controls for a language ('en' or 'ar').

    Both corpora are stored clean, so nothing is transformed here -- the
    Arabic one had lam-alef ligatures swapped by the original PDF extraction
    and was repaired in the file itself. (Evidence the USER uploads is still
    repaired at runtime in evidence_pipeline, since customer Arabic PDFs
    routinely carry the same defect.) Results are cached: the corpora are
    static and parsing them on every request is wasted work.
    """
    if language in _CONTROLS_CACHE:
        return _CONTROLS_CACHE[language]

    path = CONTROLS_PATH if language == "en" else ENGINE_DIR / "ingestion" / "controls_ar.jsonl"
    with open(path, encoding="utf-8") as f:
        controls = [json.loads(line) for line in f if line.strip()]

    _CONTROLS_CACHE[language] = controls
    return controls


# Keep the judge prompt bounded so a whole batch stays a comfortable size.
MAX_EXCERPTS_PER_JUDGEMENT = 8
MAX_EXCERPT_CHARS = 1200


def _parse_verdict(raw: dict) -> dict:
    status = str(raw.get("status_code", "")).upper()
    if status not in STATUS_LABELS:
        raise ValueError(f"LLM returned invalid status_code: {status!r}")
    try:
        confidence = max(0, min(100, int(raw.get("confidence_score", 0))))
    except (TypeError, ValueError):
        confidence = 0
    return {
        "status_code": status,
        "justification": str(raw.get("justification", "")).strip(),
        "recommendation": str(raw.get("recommendation", "")).strip(),
        "confidence_score": confidence,
    }


def _judge_batch(batch: list[dict], language: str = "en") -> dict[str, dict]:
    """Judge a batch of controls in ONE LLM request.

    `batch`: [{"control": <control dict>, "excerpts": [<chunk dict>, ...]}].
    `language`: "ar" makes the judge write its prose fields in Arabic.
    Returns {control_id: verdict dict}.
    """
    import llm_client

    blocks = []
    for item in batch:
        control = item["control"]
        excerpt_blocks = [
            f"[source: {e['source']} | type: {e['evidence_type']}]\n{e['text'][:MAX_EXCERPT_CHARS]}"
            for e in item["excerpts"][:MAX_EXCERPTS_PER_JUDGEMENT]
        ]
        blocks.append(
            f"### Control {control['control_id']}\n{control['text']}\n\n"
            "#### Evidence excerpts for this control\n"
            + "\n---\n".join(excerpt_blocks)
        )
    user_prompt = "\n\n".join(blocks)
    expected_ids = {item["control"]["control_id"] for item in batch}

    system_prompt = JUDGE_SYSTEM_PROMPT
    if language == "ar":
        system_prompt += ARABIC_OUTPUT_INSTRUCTION

    last_error = None
    for attempt in range(2):
        try:
            text, model = llm_client.generate_json(
                contents=user_prompt,
                system_instruction=system_prompt,
                preferred_model=JUDGE_MODEL,
                return_model=True,
            )
            print(f"[judge] controls {sorted(expected_ids)} judged by {model}", flush=True)
            data = json.loads(text)
            missing = expected_ids - set(data)
            if missing:
                raise ValueError(f"LLM response missing controls: {sorted(missing)}")
            return {cid: _parse_verdict(data[cid]) for cid in expected_ids}
        except RuntimeError:
            raise  # whole fallback chain exhausted -- no point retrying
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc  # malformed/incomplete response -> one more try
    raise RuntimeError(f"LLM judge returned unusable output: {last_error}")


def run_evidence_compliance_check(evidence_chunks: list[dict], language: str = "en") -> list[dict]:
    """Grade every SAMA CSF control against a multi-file evidence package.

    `evidence_chunks`: [{"text", "source", "evidence_type", "category",
    "description", "related_control"}] as produced by
    evidence_pipeline.build_evidence_chunks().

    Returns result dicts in the agreed report schema (control_id,
    control_domain, control_text, status_code, status_label,
    matched_policy_excerpt, justification, recommendation) plus the additive
    fields evidence_source (list of filenames) and confidence_score (0-100).
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a key at "
            "https://platform.openai.com/api-keys and add it to the .env file in "
            "the project root (OPENAI_API_KEY=sk-...) before running a compliance check."
        )

    controls = _load_controls(language)

    # Hybrid index over the evidence package: Chroma vector search + BM25
    # keyword search fused with RRF (see engine/retrieval.py).
    retriever = HybridRetriever([c["text"] for c in evidence_chunks])

    # Pass 1: retrieval only (cheap, local, sequential is fine). Decide per
    # control whether it needs an LLM judgement or is an automatic FAIL.
    progress.update(stage="retrieving", detail="", current=0, total=len(controls))
    prepared = []
    for control in controls:
        query_text = f"{control['title']}. {control['principle']} {control['objective']}"
        top_idx, sims = retriever.search(query_text, TOP_K_CHUNKS)

        # Threshold decisions stay on the semantic similarity scale: BM25
        # scores are corpus-relative and unbounded, so they can't be compared
        # against a fixed cutoff the way cosine similarity can.
        best_score = float(sims.max()) if len(sims) else 0.0

        # Source-aware retrieval: every submitted evidence file was provided
        # for a reason, so include each file's single best chunk too (if it
        # clears the relevance threshold), even when another file dominates
        # the global top-k. Keeps screenshots/configs from being crowded out
        # by a wordy policy document.
        selected = set(top_idx)
        best_by_source: dict[str, int] = {}
        for i, chunk in enumerate(evidence_chunks):
            src = chunk["source"]
            if src not in best_by_source or sims[i] > sims[best_by_source[src]]:
                best_by_source[src] = i
        for src, i in best_by_source.items():
            if i not in selected and float(sims[i]) >= SOURCE_FAIRNESS_THRESHOLD:
                top_idx.append(i)
                selected.add(i)

        excerpts = [evidence_chunks[i] for i in top_idx]

        # Evidence the user explicitly tagged with this control is always
        # included, even if embedding similarity missed it.
        pinned = [
            c for c in evidence_chunks
            if c["related_control"] and c["related_control"] == control["control_id"]
            and c not in excerpts
        ]
        if pinned:
            excerpts = pinned + excerpts
            best_score = max(best_score, NO_MATCH_THRESHOLD)

        prepared.append({"control": control, "excerpts": excerpts, "best_score": best_score})
        progress.advance()

    # Pass 2: judge every control that has relevant evidence, several
    # controls per LLM request, batches sent concurrently. Controls with
    # no relevant evidence at all skip the LLM entirely (automatic FAIL).
    to_judge = [item for item in prepared if item["best_score"] >= NO_MATCH_THRESHOLD]
    batches = [
        to_judge[i : i + JUDGE_BATCH_SIZE]
        for i in range(0, len(to_judge), JUDGE_BATCH_SIZE)
    ]

    verdicts: dict[str, dict] = {}
    progress.update(stage="judging", detail="", current=0, total=len(batches))
    if batches:
        with ThreadPoolExecutor(max_workers=min(JUDGE_CONCURRENCY, len(batches))) as pool:
            futures = {pool.submit(_judge_batch, batch, language): batch for batch in batches}
            for future in as_completed(futures):
                batch = futures[future]
                batch_ids = [item["control"]["control_id"] for item in batch]
                try:
                    verdicts.update(future.result())
                    progress.advance()
                except RuntimeError as exc:
                    # Surface which controls' batch caused the failure.
                    raise RuntimeError(f"{exc} (controls {batch_ids})") from exc

    # Pass 3: assemble results in the controls' original order.
    results = []
    for item in prepared:
        control = item["control"]
        excerpts = item["excerpts"]

        if item["best_score"] < NO_MATCH_THRESHOLD:
            # No LLM call is made here, so these strings are localized by hand.
            if language == "ar":
                verdict = {
                    "status_code": "FAIL",
                    "justification": (
                        "لم يُعثر في حزمة الأدلة المقدمة على أي محتوى ذي صلة "
                        "يعالج متطلبات هذا الضابط."
                    ),
                    "recommendation": (
                        f"يجب تقديم أدلة تغطي متطلبات '{control['title']}' وفقاً "
                        "لإطار الأمن السيبراني الصادر عن البنك المركزي السعودي "
                        "(قسم من السياسة، أو تصدير للإعدادات، أو لقطة شاشة تُثبت "
                        "تطبيق الضابط)."
                    ),
                    "confidence_score": 90,
                }
                matched_excerpt = "لم يُعثر على محتوى مطابق في الأدلة المقدمة."
            else:
                verdict = {
                    "status_code": "FAIL",
                    "justification": (
                        "No relevant content addressing this control was found in the "
                        "submitted evidence package."
                    ),
                    "recommendation": (
                        f"Provide evidence covering '{control['title']}' requirements as "
                        "defined by SAMA CSF (policy section, configuration export, or "
                        "screenshot demonstrating the control)."
                    ),
                    "confidence_score": 90,
                }
                matched_excerpt = "No matching content found in the submitted evidence."
            evidence_source: list[str] = []
        else:
            verdict = verdicts[control["control_id"]]
            matched_excerpt = excerpts[0]["text"]
            # Preserve retrieval order but deduplicate filenames.
            evidence_source = list(dict.fromkeys(e["source"] for e in excerpts))

        results.append(
            {
                "control_id": control["control_id"],
                "control_domain": control["domain"],
                "control_text": control["text"],
                "status_code": verdict["status_code"],
                "status_label": STATUS_LABELS[verdict["status_code"]],
                "matched_policy_excerpt": matched_excerpt,
                "justification": verdict["justification"],
                "recommendation": verdict["recommendation"],
                "evidence_source": evidence_source,
                "confidence_score": verdict["confidence_score"],
            }
        )
    return results
