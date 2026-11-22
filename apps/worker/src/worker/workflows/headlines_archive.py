"""Headlines archive workflow: long-lived cross-workflow memory of daily headlines.

The daily news brief signals record_day(date, headlines) each morning; the
weekly deep-dive signals request_recent(requester_id, days) and receives the
recent entries back via a receive_recent signal, so it can avoid re-reporting
stories the dailies already covered.

The workflow never returns: it sleeps a week, prunes entries older than
KEEP_DAYS, and continues as new (keeping history bounded and the event log
small). Signals reach workflows indefinitely, so replays after
continue-as-new keep working with the same workflow id.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import mistralai.workflows as workflows
from temporalio import workflow as _tw
from pydantic import BaseModel, Field

HEADLINES_ARCHIVE_WORKFLOW_ID = "headlines-archive-v1"
RECORD_DAY_SIGNAL = "record_day"
REQUEST_RECENT_SIGNAL = "request_recent"
RECEIVE_RECENT_SIGNAL = "receive_recent"

KEEP_DAYS = 30
MAX_HEADLINES_PER_DAY = 40
MAX_HEADLINE_LEN = 160
MAX_ENTRIES = 60


class ArchiveInput(BaseModel):
    """Input for the headlines archive (entries carried across continue-as-new)."""

    entries: list[dict] = Field(default_factory=list)


def _cutoff_date(days: int) -> str:
    return (_tw.now().date() - timedelta(days=days)).isoformat()


def _clean_entry(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    date = str(raw.get("date", ""))[:10]
    raw_headlines = raw.get("headlines") or []
    if not date:
        return None
    headlines = [str(h)[:MAX_HEADLINE_LEN] for h in raw_headlines][:MAX_HEADLINES_PER_DAY]
    return {"date": date, "headlines": headlines}


@workflows.workflow.define(
    name="headlines_archive",
    execution_timeout=timedelta(days=365),
)
class HeadlinesArchiveWorkflow:
    _entries: list[dict] = []

    def _prune(self, days: int = KEEP_DAYS) -> None:
        cutoff = _cutoff_date(days)
        entries = []
        for raw in self._entries:
            entry = _clean_entry(raw)
            if entry and entry["date"] >= cutoff:
                entries.append(entry)
        entries.sort(key=lambda e: e["date"])
        self._entries = entries[-MAX_ENTRIES:]

    @workflows.workflow.signal(name=RECORD_DAY_SIGNAL)
    async def record_day(self, date: str, headlines: list) -> None:
        """Replace (or add) the entry for a date. Survives continue-as-new."""
        entry = _clean_entry({"date": date, "headlines": headlines or []})
        if entry is None:
            return
        kept = [e for e in self._entries if e.get("date") != entry["date"]]
        kept.append(entry)
        self._entries = kept
        self._prune()

    @workflows.workflow.signal(name=REQUEST_RECENT_SIGNAL)
    async def request_recent(self, requester_id: str, days: int = 7) -> None:
        """Reply to the requester with entries from the last `days` days."""
        cutoff = _cutoff_date(days)
        recent = [e for e in self._entries if e.get("date", "") >= cutoff]
        try:
            handle = _tw.get_external_workflow_handle(requester_id)
            await handle.signal(RECEIVE_RECENT_SIGNAL, recent)
        except Exception:
            pass  # requester already finished or unreachable: ignore

    @workflows.workflow.entrypoint
    async def run(self, data: ArchiveInput) -> dict:
        self._entries = list(data.entries or [])
        self._prune()
        while True:
            await _tw.sleep(timedelta(days=7))
            self._prune()
            _tw.continue_as_new({"entries": self._entries})
