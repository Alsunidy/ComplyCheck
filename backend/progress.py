"""
ComplyCheck - live progress tracking for long-running work.

/upload-evidence and /run-compliance-check are synchronous calls that can take
tens of seconds (vision OCR per screenshot, 36 controls judged in batches).
A spinner that only says "working..." gives the user no idea whether the
system is stuck or nearly done, so the pipeline reports each stage here and
the UI polls GET /progress to show what is actually happening.

State is a single in-process dict: this is a single-user local demo, matching
the existing EVIDENCE_PACKAGE / REPORTS_DB approach. A multi-user deployment
would key this by job id.
"""
import threading
import time

_lock = threading.Lock()

# stage: machine-readable step name the UI maps to a localized label.
# detail: free text (a filename, a control id range) shown beside it.
# current/total: populated for countable work (batches, files).
_state: dict = {
    "stage": "idle",
    "detail": "",
    "current": 0,
    "total": 0,
    "started_at": None,
    "done": True,
}

# Stage names set by the pipeline, in the order they occur:
#   uploading -> classifying -> extracting / analyzing_images ->
#   detecting_language -> indexing -> retrieving -> judging -> complete
# The UI maps them to localized labels (frontend/ui_text.py STAGE_LABELS).


def start(stage: str, detail: str = "", total: int = 0) -> None:
    with _lock:
        _state.update(
            stage=stage, detail=detail, current=0, total=total,
            started_at=time.time(), done=False,
        )


def update(stage: str | None = None, detail: str | None = None,
           current: int | None = None, total: int | None = None) -> None:
    """Set whichever fields are provided; leave the rest untouched."""
    with _lock:
        if stage is not None:
            _state["stage"] = stage
        if detail is not None:
            _state["detail"] = detail
        if current is not None:
            _state["current"] = current
        if total is not None:
            _state["total"] = total
        _state["done"] = False


def advance(step: int = 1) -> None:
    """Increment the counter for countable work (one batch/file finished)."""
    with _lock:
        _state["current"] = _state.get("current", 0) + step


def finish(detail: str = "") -> None:
    with _lock:
        _state.update(stage="complete", detail=detail, done=True)


def snapshot() -> dict:
    with _lock:
        state = dict(_state)
    started = state.get("started_at")
    state["elapsed"] = round(time.time() - started, 1) if started else 0.0
    return state
