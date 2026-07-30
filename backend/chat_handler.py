"""
ComplyCheck - SAMA Compliance Auditor
Chat / Q&A, grounded by retrieval and answered by the LLM.

Two modes, chosen by whether a compliance report exists yet:

  1. No report (available from the moment the app opens): the question is
     answered from the SAMA CSF corpus itself. The 36 controls are indexed
     once in the same hybrid retriever the audit engine uses (Chroma vector
     search + BM25 fused with RRF), so users can ask "what does SAMA require
     for incident management?" before uploading anything.

  2. Report available: the user's own results become the grounding context --
     scoped to one control when control_id is given, otherwise the whole
     report -- so questions like "which controls need urgent attention?" are
     answered from their actual audit.

Follow-up questions are handled by rewriting them into a standalone query
before retrieval (see _rewrite_query): "elaborate more" shares no words with
the topic under discussion, so searching it literally returns the wrong
control.

In both modes the LLM is instructed to answer ONLY from the retrieved
context and to say it doesn't have the information rather than invent it.
Answers follow the language of the question (or of the report, when one
exists).
"""
import json
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent / "engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import arabic_text  # noqa: E402
import llm_client  # noqa: E402

# How many SAMA controls to retrieve as context for a framework question.
TOP_K_CONTROLS = 5
# Turns of prior conversation sent to the model. Enough for follow-ups to
# make sense without letting the prompt grow unbounded.
MAX_HISTORY_MESSAGES = 6

# Follow-up questions are meaningless to a retriever on their own: "elaborate
# more" or "فصّل أكثر" share no words with the topic being discussed, so the
# search matches whatever those words literally mean (a real example: "فصل
# اكثر" retrieved the control about SEPARATING the security unit from IT,
# because "فصل" means separation). This rewrites the latest message into a
# standalone query using the conversation, before retrieval runs.
REWRITE_PROMPT = (
    "You rewrite the user's latest message into a standalone search query for "
    "a SAMA Cyber Security Framework knowledge base.\n\n"
    "Use the conversation to resolve references: pronouns, and vague "
    'follow-ups such as "tell me more", "elaborate", "and the second one?", '
    '"فصّل أكثر", "اشرح أكثر". The rewritten query must name the actual '
    "subject being discussed, in the SAME language as the user's message. If "
    "the latest message is already self-contained, return it unchanged.\n\n"
    'Respond with JSON: {"query": "the standalone search query"}'
)

SYSTEM_PROMPT = (
    "You are the ComplyCheck compliance assistant, helping a GRC team with "
    "the SAMA Cyber Security Framework (CSF).\n\n"
    "Answer using ONLY the context provided in the user message (SAMA CSF "
    "control texts and, when present, the organization's own compliance "
    "assessment results). Do not use outside knowledge and do not invent "
    "control requirements, statuses, or evidence. If the context does not "
    "contain what is needed, say: \"I don't have that information in the "
    "provided context.\"\n\n"
    "Be direct and specific: quote control IDs, statuses and concrete "
    "requirements rather than speaking generally. Keep answers concise "
    "(under ~150 words) unless the user asks for detail.\n\n"
    "Respond with a JSON object with exactly these keys:\n"
    '  "answer": your reply as a string (plain text, may use "- " bullets)\n'
    '  "cited_control_ids": array of the control_id strings you actually '
    "used; empty array if none apply"
)

# Appended when the user writes in Arabic: the corpus, the answer and the
# report all follow the language the user is working in.
ARABIC_REPLY_INSTRUCTION = (
    "\n\nThe user is writing in Arabic and the context is Arabic. Write the "
    '"answer" value in formal Modern Standard Arabic. Keep the JSON keys and '
    "the control_id values exactly as they are."
)

# The SAMA controls corpora are static, so each language's hybrid index is
# built once and reused for every chat request in that language.
_retriever_cache: dict = {}


def _load_controls(language: str = "en") -> list[dict]:
    import compliance_engine

    return compliance_engine._load_controls(language)


def _control_to_text(control: dict) -> str:
    return (
        f"[SAMA CSF control {control['control_id']} -- {control['title']} "
        f"({control['domain']})]\n{control['text']}"
    )


def _get_controls_retriever(language: str = "en"):
    if language not in _retriever_cache:
        from retrieval import HybridRetriever

        controls = _load_controls(language)
        _retriever_cache[language] = HybridRetriever(
            [_control_to_text(c) for c in controls]
        )
    return _retriever_cache[language]


def _find_control(report: dict, control_id: str) -> dict | None:
    for result in report.get("results", []):
        if result["control_id"] == control_id:
            return result
    return None


def _result_to_text(result: dict) -> str:
    return (
        f"[Assessment of control {result['control_id']} -- "
        f"{result['control_domain']}]\n"
        f"requirement: {result['control_text']}\n"
        f"status: {result['status_label']}\n"
        f"evidence excerpt: {result['matched_policy_excerpt']}\n"
        f"evidence sources: {', '.join(result.get('evidence_source') or []) or 'none'}\n"
        f"justification: {result['justification']}\n"
        f"recommendation: {result['recommendation']}"
    )


def _format_history(history: list[dict] | None) -> str:
    if not history:
        return ""
    recent = history[-MAX_HISTORY_MESSAGES:]
    lines = [
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
        for m in recent
        if m.get("content")
    ]
    return "\n".join(lines)


def _rewrite_query(question: str, history: list[dict] | None) -> str:
    """Turn a follow-up into a standalone retrieval query."""
    if not history:
        return question
    conversation = _format_history(history)
    if not conversation:
        return question
    try:
        raw = llm_client.generate_json(
            contents=(
                f"## Conversation\n{conversation}\n\n"
                f"## Latest message\n{question}"
            ),
            system_instruction=REWRITE_PROMPT,
        )
        rewritten = str(json.loads(raw).get("query", "")).strip()
        return rewritten or question
    except Exception:
        # Rewriting is an optimisation; never fail the answer because of it.
        return question


def _build_context(question: str, report: dict | None, control_id: str | None,
                   language: str = "en") -> tuple[str, list[str]]:
    """Return (context text for the LLM, control_ids available in context)."""
    if report and control_id:
        result = _find_control(report, control_id)
        if result is None:
            return "", []
        return _result_to_text(result), [control_id]

    if report:
        results = report.get("results", [])
        summary = (
            f"The report assessed {len(results)} SAMA CSF controls: "
            f"{sum(1 for r in results if r['status_code'] == 'PASS')} compliant, "
            f"{sum(1 for r in results if r['status_code'] == 'PARTIAL')} partially compliant, "
            f"{sum(1 for r in results if r['status_code'] == 'FAIL')} non-compliant.\n"
            f"Evidence files used: "
            f"{', '.join(e['filename'] for e in report.get('evidence_used', [])) or 'unknown'}"
        )
        blocks = [summary] + [_result_to_text(r) for r in results]
        return "\n\n".join(blocks), [r["control_id"] for r in results]

    # No report yet -- ground the answer in the SAMA CSF corpus of the
    # language the user is asking in.
    controls = _load_controls(language)
    retriever = _get_controls_retriever(language)
    top_idx, _ = retriever.search(question, TOP_K_CONTROLS)
    selected = [controls[i] for i in top_idx]
    blocks = [_control_to_text(c) for c in selected]
    return "\n\n".join(blocks), [c["control_id"] for c in selected]


def answer_report_question(
    question: str,
    report: dict | None = None,
    control_id: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    """Answer a question about the SAMA CSF or about a compliance report.

    `history` is the prior conversation as [{"role", "content"}], used both to
    resolve follow-up questions into standalone retrieval queries and to give
    the model conversational context.

    Returns {"answer": str, "cited_control_ids": list[str]}.
    """
    # The report's language wins when one exists (the answer discusses that
    # report); otherwise follow the language of the question itself.
    language = (report or {}).get("language") or arabic_text.detect_language(question)
    if language not in ("en", "ar"):
        language = "en"

    # Retrieve with a standalone version of the question; answer the original.
    search_query = _rewrite_query(question, history)
    context, available_ids = _build_context(search_query, report, control_id, language)

    if not context:
        return {
            "answer": (
                "لا تتوفر لديّ هذه المعلومة ضمن السياق المتاح."
                if language == "ar"
                else "I don't have that information in the provided context."
            ),
            "cited_control_ids": [],
        }

    scope = (
        "The user has not run a compliance check yet, so the context below is "
        "the SAMA CSF requirements themselves."
        if report is None
        else "The context below is the organization's own compliance assessment."
    )
    # The prior turns let the model answer conversationally ("elaborate on
    # that") instead of treating every message as a fresh question.
    conversation = _format_history(history)
    history_block = f"## Conversation so far\n{conversation}\n\n" if conversation else ""
    user_prompt = (
        f"{scope}\n\n{history_block}## Context\n{context}\n\n## Question\n{question}"
    )

    system_prompt = SYSTEM_PROMPT
    if language == "ar":
        system_prompt += ARABIC_REPLY_INSTRUCTION

    try:
        raw = llm_client.generate_json(
            contents=user_prompt, system_instruction=system_prompt
        )
        data = json.loads(raw)
    except RuntimeError as exc:
        return {"answer": f"The assistant is temporarily unavailable: {exc}", "cited_control_ids": []}
    except (json.JSONDecodeError, ValueError):
        return {
            "answer": (
                "تعذّر إنتاج إجابة موثوقة لهذا السؤال. يُرجى إعادة صياغته."
                if language == "ar"
                else "I couldn't produce a reliable answer for that question. Please rephrase it."
            ),
            "cited_control_ids": [],
        }

    answer = str(data.get("answer", "")).strip()
    if not answer:
        answer = (
            "لا تتوفر لديّ هذه المعلومة ضمن السياق المتاح."
            if language == "ar"
            else "I don't have that information in the provided context."
        )

    # Keep only citations that really exist in the context we supplied.
    cited = [
        str(cid) for cid in (data.get("cited_control_ids") or [])
        if str(cid) in set(available_ids)
    ]
    return {"answer": answer, "cited_control_ids": cited}
