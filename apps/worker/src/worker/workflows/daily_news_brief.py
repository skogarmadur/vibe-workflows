"""Daily news brief workflow: UK, Ireland, Sweden, Denmark, France.

Fetches headlines from domestic-language RSS feeds for the last 24 hours,
then writes Guðmundur's morning brief via several small Mistral model calls
(one per country, one for the dedicated woodlands/forestry slot, plus a
short overview pass). The platform HTTP client read-times-out model calls
that take too long (~5 minutes), so no single call may cover every feed at
once, and each section is its own activity so a retry only replays one
small call.

Designed for an unattended scheduled run: reads only public feeds, no
user-scope credentials. Deliverable is the execution result (read in Studio).

After the brief is assembled, the day's story headlines are pushed to the
long-lived headlines_archive workflow (best-effort): the weekly deep-dive
reads that archive to avoid re-reporting stories the dailies already
covered. The archive push NEVER fails the daily.
"""

from __future__ import annotations

from datetime import timedelta
import re
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import temporalio.common as _tcommon
from temporalio import workflow as _tw

import mistralai.workflows as workflows
from mistralai.workflows.plugins.mistralai import (
    ChatCompletionRequest,
    UserMessage,
    mistralai_chat_complete,
)
from pydantic import BaseModel, Field

from worker.workflows.headlines_archive import (
    HEADLINES_ARCHIVE_WORKFLOW_ID,
    HeadlinesArchiveWorkflow,
    RECORD_DAY_SIGNAL,
)

FEEDS: tuple[tuple[str, str, str], ...] = (
    ("UK", "BBC News", "https://feeds.bbci.co.uk/news/uk/rss.xml"),
    ("UK", "The Guardian", "https://www.theguardian.com/uk/rss"),
    ("Ireland", "RTÉ", "https://www.rte.ie/feeds/rss/?index=/news/"),
    ("Ireland", "TheJournal.ie", "https://www.thejournal.ie/feed/"),
    ("Sweden", "SVT Nyheter", "https://www.svt.se/nyheter/rss.xml"),
    ("Sweden", "Dagens Nyheter", "https://www.dn.se/nyheter/rss/"),
    ("Denmark", "Politiken", "https://politiken.dk/rss/senestenyt.rss"),
    ("Denmark", "Ekstra Bladet", "https://ekstrabladet.dk/rssfeed/all/"),
    ("France", "Le Monde", "https://www.lemonde.fr/rss/une.xml"),
    ("France", "franceinfo", "https://www.francetvinfo.fr/titres.rss"),
    ("Topics", "The Guardian (Environment)", "https://www.theguardian.com/environment/rss"),
)

COUNTRIES = ("UK", "Ireland", "Sweden", "Denmark", "France")

WOODLAND_EMPTY_LINE = (
    "Nothing of note today on woodlands, forestry or woodland restoration "
    "across the covered countries."
)

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_BOLD_HEADLINE_RE = re.compile(r"^\*\*(.+?)\*\*", re.MULTILINE)


def _clean(text: str | None) -> str:
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _text(element: ET.Element, tag: str) -> str:
    found = element.find(tag)
    if found is not None and found.text:
        return found.text
    # Some feeds wrap text in CDATA or nested elements.
    for descendant in element.iter(tag):
        if descendant.text:
            return descendant.text
    return None


def _extract_text(content) -> str:
    if isinstance(content, list):
        # Model replies may arrive as content blocks (thinking + text):
        # keep ONLY the text of each block; skip thinking content entirely.
        parts = []
        for block in content:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "\n\n".join(parts)
    return content or ""


def _story_headlines(markdown_text: str) -> list[str]:
    """Extract bold story headlines from assembled section markdown."""
    out: list[str] = []
    for h in _BOLD_HEADLINE_RE.findall(markdown_text):
        h = h.strip().rstrip(".").strip()
        if h and len(h) > 12 and h not in out:
            out.append(h[:160])
    return out[:40]


def _daily_date() -> str:
    """Date string (Atlantic/Reykjavik) for the archive entry. Deterministic."""
    now = _tw.now()
    try:
        return now.astimezone(ZoneInfo("Atlantic/Reykjavik")).date().isoformat()
    except Exception:
        return now.date().isoformat()


async def _record_daily_headlines(date: str, headlines: list[str]) -> bool:
    """Best-effort push to the running headlines archive. Never raises.

    Starts the archive workflow if it is not running yet (id-reuse ALLOW_DUPLICATE
    makes this a no-op when it already runs), then signals record_day. Any
    failure is swallowed: the daily brief must never break because of the
    archive.
    """
    if not date or not headlines:
        return False
    try:
        try:
            # Will raise if the archive is already running: the normal case
            # from the second day onwards.
            await _tw.start_child_workflow(
                workflows.get_workflow_definition(HeadlinesArchiveWorkflow).name,
                {"entries": []},
                id=HEADLINES_ARCHIVE_WORKFLOW_ID,
                id_reuse_policy=_tcommon.WorkflowIDReusePolicy.ALLOW_DUPLICATE,
                parent_close_policy=_tw.ParentClosePolicy.ABANDON,
            )
        except Exception:
            pass
        handle = _tw.get_external_workflow_handle(HEADLINES_ARCHIVE_WORKFLOW_ID)
        await handle.signal(RECORD_DAY_SIGNAL, {"date": date, "headlines": headlines})
        return True
    except Exception:
        return False


class NewsBriefInput(BaseModel):
    """Input for the daily news brief."""

    max_items_per_feed: int = Field(default=10, ge=1, le=30)
    model: str = "mistral-large-latest"
    dry_run: bool = False


class NewsBriefOutput(BaseModel):
    brief_markdown: str
    feeds_ok: list[str]
    feeds_failed: list[str]
    item_count: int
    model: str
    archived_headlines: int = 0


@workflows.activity(
    start_to_close_timeout=timedelta(seconds=90),
    retry_policy_max_attempts=2,
)
async def fetch_feed(country: str, source: str, url: str, max_items: int) -> list[dict]:
    """Fetch and parse one public RSS feed. Failures return [] (handled upstream)."""
    import httpx

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(15.0),
        headers={"User-Agent": "Mozilla/5.0 (compatible; news-brief-workflow/1.0)"},
        follow_redirects=True,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.text

    items: list[dict] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return items
    for element in root.iter("item"):
        title = _clean(_text(element, "title"))
        link = (_text(element, "link") or "").strip()
        summary = _clean(_text(element, "description"))
        if title and link:
            items.append(
                {
                    "country": country,
                    "source": source,
                    "title": title,
                    "link": link,
                    "summary": summary[:600],
                }
            )
        if len(items) >= max_items:
            break
    return items


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=6),
    retry_policy_max_attempts=3,
)
async def write_country_section(country: str, feed_items: list[dict], topics_items: list[dict], model: str) -> dict:
    """Write one country's section via a single small model call.

    Kept small deliberately: the platform HTTP client read-times-out long
    model calls, and each country is its own activity so a retry only
    replays one small call. Returns the section plus machine-readable
    top stories for the overview pass.
    """
    from datetime import datetime

    today = datetime.now().strftime("%A %-d %B %Y")

    lines = [f"## {country}"]
    for item in feed_items:
        summary = f" — {item['summary']}" if item.get("summary") else ""
        lines.append(f"- {item['title']} ({item['source']}){summary}\n  {item['link']}")
    items_text = "\n".join(lines)

    topics_text = ""
    if topics_items:
        tlines = []
        for item in topics_items:
            summary = f" — {item['summary']}" if item.get("summary") else ""
            tlines.append(f"- {item['title']} ({item['source']}){summary}\n  {item['link']}")
        topics_text = (
            "Additional cross-border environment feed (Guardian Environment; "
            "fold any relevant items into this section if they concern this "
            "country or are otherwise significant):\n\n" + "\n".join(tlines)
        )

    prompt = (
        f"You are writing the {country} section of Guðmundur's daily morning news "
        f"brief for **{today}**, from news of the last 24 hours.\n\n"
        "Rules:\n"
        "- Begin every story with its headline in bold, followed by an em dash and "
        "the story, e.g. \"**Headline** — what happened and why it matters...\".\n"
        "- For each story: what happened, and why it matters — background, implications, "
        "key reactions and stakeholders. For developing stories, note what to watch next.\n"
        "- Sources with links are MANDATORY: after every story, on its own line, add the "
        "source link(s) as Markdown, e.g. \"[BBC News](https://...)\". If several fetched "
        "items informed the story, list them all. Never present a story without at least "
        "one link.\n"
        "- Translate Swedish, Danish and French headlines and summaries into English.\n"
        "- Depth over breadth: pick the most significant stories and treat them thoroughly; "
        "this is a thorough but digestible morning read, not an exhaustive report.\n"
        "- Hard length cap: at most the four most significant stories, and keep the "
        "whole section under 500 words.\n"
        "- Guðmundur has a standing interest in forestry, woodland restoration and "
        "conservation. Forestry and woodland stories are covered in a separate dedicated "
        "section of the brief: do NOT include them here, so the slots stay free for other "
        "news. Other environment stories may be included if significant.\n"
        "- Guðmundur does not care for sports. Omit routine sports news (match results, "
        "transfers, league tables, fixtures). Include a sports story ONLY if it is "
        "genuinely big news with significance well beyond sport, such as a major "
        "national event, scandal or tragedy. When in doubt, leave it out.\n"
        "- If the country has no significant news, say so briefly.\n"
        "- Write in UK English (colour, organise, realise).\n"
        "- If a headline's summary is thin, present the story cautiously from the headline "
        "alone rather than inventing detail.\n"
        "- Write ONLY this country's section, starting with a '## Country' heading. "
        "No overall overview; that is written separately.\n"
        "- End your reply with exactly this machine-readable block (nothing after it):\n"
        "<<<TOP>>>\n"
        "- <one-line summary of this country's biggest story>\n"
        "- <one-line summary of the second biggest, if any>\n\n"
        "Fetched headlines (last 24 hours):\n\n" + items_text + (
            "\n\n" + topics_text if topics_text else ""
        )
    )

    request = ChatCompletionRequest(
        model=model,
        messages=[UserMessage(content=prompt)],
    )
    result = await mistralai_chat_complete(request)
    content = _extract_text(result.choices[0].message.content)

    top_stories: list[str] = []
    section = content
    if "<<<TOP>>>" in content:
        section, rest = content.split("<<<TOP>>>", 1)
        top_stories = [
            l.strip()[2:].strip()
            for l in rest.strip().splitlines()
            if l.strip().startswith("-")
        ]
    return {
        "country": country,
        "section_markdown": section.strip(),
        "top_stories": top_stories,
    }


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=6),
    retry_policy_max_attempts=3,
)
async def write_woodland_section(topics_items: list[dict], model: str) -> str:
    """Write the dedicated woodlands/forestry section via one small model call.

    Guaranteed slot for Guðmundur's standing interest in forestry, woodland
    restoration and conservation. If there is nothing of worth in the last
    24 hours, the section says so in one line rather than being silently
    dropped.
    """
    if not topics_items:
        return f"## Woodlands & Forestry\n\n{WOODLAND_EMPTY_LINE}"

    from datetime import datetime

    today = datetime.now().strftime("%A %-d %B %Y")

    tlines = []
    for item in topics_items:
        summary = f" — {item['summary']}" if item.get("summary") else ""
        tlines.append(f"- {item['title']} ({item['source']}){summary}\n  {item['link']}")
    items_text = "\n".join(tlines)

    prompt = (
        f"You are writing the woodlands & forestry section of Guðmundur's daily "
        f"morning news brief for **{today}**, from news of the last 24 hours. "
        "This is a dedicated slot for his standing interest in forestry, woodland "
        "restoration and conservation across the United Kingdom, Ireland, Sweden, "
        "Denmark and France (plus significant cross-border European stories).\n\n"
        "Rules:\n"
        "- From the environment feed items below, select only stories that are "
        "genuinely about forests, woodlands, forestry, tree planting, afforestation, "
        "reforestation or forest/woodland conservation. Stories about other "
        "environmental topics (marine, climate policy, wildlife unrelated to "
        "woodland) do NOT belong here.\n"
        "- Relevance filter: prefer stories concerning the five covered countries; "
        "include a wider European or global story only if it is genuinely "
        "significant.\n"
        "- Begin every story with its headline in bold, followed by an em dash and "
        "the story, e.g. \"**Headline** — what happened and why it matters...\".\n"
        "- For each story: what happened, and why it matters — background, "
        "implications, key reactions and stakeholders. For developing stories, note "
        "what to watch next.\n"
        "- Sources with links are MANDATORY: after every story, on its own line, add "
        "the source link(s) as Markdown, e.g. \"[The Guardian](https://...)\".\n"
        "- Length cap: at most two stories, and keep the whole section under 200 "
        "words.\n"
        "- If NOTHING in the feed is a forestry/woodland/conservation story of "
        "worth, write exactly this section and nothing else:\n"
        "## Woodlands & Forestry\n\n"
        "Nothing of note today on woodlands, forestry or woodland restoration "
        "across the covered countries.\n"
        "Do not pad with weakly related items.\n"
        "- Write in UK English (colour, organise, realise).\n"
        "- If a headline's summary is thin, present the story cautiously from the "
        "headline alone rather than inventing detail.\n"
        "- Write ONLY the section, starting with the '## Woodlands & Forestry' "
        "heading. No overview, no sign-off.\n\n"
        "Environment feed items (last 24 hours):\n\n" + items_text
    )

    request = ChatCompletionRequest(
        model=model,
        messages=[UserMessage(content=prompt)],
    )
    result = await mistralai_chat_complete(request)
    return _extract_text(result.choices[0].message.content).strip()


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=6),
    retry_policy_max_attempts=3,
)
async def write_daily_overview(top_stories_text: str, model: str) -> str:
    """Write the one-paragraph opening overview from the per-country top stories."""
    from datetime import datetime

    today = datetime.now().strftime("%A %-d %B %Y")
    prompt = (
        f"You are writing the opening paragraph of Guðmundur's daily morning news "
        f"brief for **{today}**, covering the United Kingdom, Ireland, Sweden, "
        "Denmark and France.\n\n"
        "Below are the biggest stories from each country section of the brief. "
        "Write a single opening paragraph (no heading, no bullet points) giving "
        "the big picture across the five countries and naming the biggest story "
        "overall. Write in UK English.\n\n" + top_stories_text
    )
    request = ChatCompletionRequest(
        model=model,
        messages=[UserMessage(content=prompt)],
    )
    result = await mistralai_chat_complete(request)
    return _extract_text(result.choices[0].message.content).strip()


@workflows.workflow.define(name="daily_news_brief")
class DailyNewsBriefWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, data: NewsBriefInput) -> NewsBriefOutput:
        items: list[dict] = []
        feeds_ok: list[str] = []
        feeds_failed: list[str] = []
        for country, source, url in FEEDS:
            feed_items = await fetch_feed(country, source, url, data.max_items_per_feed)
            if feed_items:
                items.extend(feed_items)
                feeds_ok.append(f"{source} ({country})")
            else:
                feeds_failed.append(f"{source} ({country})")

        if data.dry_run:
            headlines = "\n".join(f"- {i['title']} — {i['link']}" for i in items)
            return NewsBriefOutput(
                brief_markdown=(
                    f"Dry run: fetched {len(items)} items. "
                    f"OK feeds: {len(feeds_ok)}, failed: {len(feeds_failed)}.\n\n{headlines}"
                ),
                feeds_ok=feeds_ok,
                feeds_failed=feeds_failed,
                item_count=len(items),
                model="(dry run, no model call)",
            )

        grouped: dict[str, list[dict]] = {}
        for item in items:
            grouped.setdefault(item["country"], []).append(item)
        topics_items = grouped.pop("Topics", [])

        sections: list[dict] = []
        for country in COUNTRIES:
            feed_items = grouped.get(country, [])
            if not feed_items:
                sections.append(
                    {
                        "country": country,
                        "section_markdown": f"## {country}\n\nNo feeds available today.",
                        "top_stories": [],
                    }
                )
                continue
            sections.append(
                await write_country_section(country, feed_items, topics_items, data.model)
            )

        woodland_section = await write_woodland_section(topics_items, data.model)

        top_stories_text = "\n\n".join(
            f"{s['country']}:\n" + "\n".join(f"- {t}" for t in s["top_stories"])
            for s in sections
            if s["top_stories"]
        )
        overview = ""
        if top_stories_text:
            overview = await write_daily_overview(top_stories_text, data.model)

        parts = [
            p
            for p in (
                overview,
                "\n\n".join(s["section_markdown"] for s in sections),
                woodland_section,
            )
            if p
        ]
        brief = "\n\n".join(parts)

        archived = 0
        if not data.dry_run:
            headlines = _story_headlines(brief)
            if await _record_daily_headlines(_daily_date(), headlines):
                archived = len(headlines)

        return NewsBriefOutput(
            brief_markdown=brief,
            feeds_ok=feeds_ok,
            feeds_failed=feeds_failed,
            item_count=len(items),
            model=data.model,
            archived_headlines=archived,
        )
