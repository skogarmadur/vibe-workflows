"""Daily news brief workflow: UK, Ireland, Sweden, Denmark, France.

Fetches headlines from domestic-language RSS feeds for the last 24 hours,
then writes Guðmundur's morning brief format via a Mistral model call.
Designed for an unattended scheduled run: reads only public feeds, no
user-scope credentials. Deliverable is the execution result (read in Studio).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import mistralai.workflows as workflows
from mistralai.workflows.plugins.mistralai import (
    ChatCompletionRequest,
    UserMessage,
    mistralai_chat_complete,
)
from pydantic import BaseModel, Field

# Public RSS feeds, verified reachable 2026-11-12. One feed per (country, source).
# "Topics" feeds cover standing interests that cross borders.
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

COUNTRIES = ("UK", "Ireland", "Sweden", "Denmark", "France", "Topics")

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean(text: str | None) -> str:
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _text(element: ET.Element, tag: str) -> str | None:
    found = element.find(tag)
    if found is not None and found.text:
        return found.text
    # Some feeds wrap text in CDATA or nested elements.
    for descendant in element.iter(tag):
        if descendant.text:
            return descendant.text
    return None


class NewsBriefInput(BaseModel):
    """Input for the daily news brief."""

    max_items_per_feed: int = Field(default=10, ge=1, le=30)
    model: str = "mistral-large-latest"
    dry_run: bool = False


class FeedItem(BaseModel):
    country: str
    source: str
    title: str
    link: str
    summary: str


class NewsBriefOutput(BaseModel):
    brief_markdown: str
    feeds_ok: list[str]
    feeds_failed: list[str]
    item_count: int
    model: str


@workflows.activity()
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


@workflows.activity()
async def write_brief(feed_items: list[dict], model: str) -> str:
    """Call a Mistral model to write the brief from the fetched headlines."""
    from datetime import datetime

    # The current date is computed in the activity (allowed: activities may
    # use non-deterministic sources) so the model never has to guess it.
    today = datetime.now().strftime("%A %-d %B %Y")

    grouped: dict[str, list[dict]] = {country: [] for country in COUNTRIES}
    for item in feed_items:
        grouped.setdefault(item["country"], []).append(item)

    sections: list[str] = []
    for country in COUNTRIES:
        lines = [f"## {country}"]
        for item in grouped[country]:
            summary = f" — {item['summary']}" if item.get("summary") else ""
            lines.append(f"- {item['title']} ({item['source']}){summary}\n  {item['link']}")
        sections.append("\n".join(lines))

    prompt = (
        f"You are writing Guðmundur's daily morning news brief for **{today}**, "
        "covering the United Kingdom, Ireland, Sweden, Denmark and France, from news "
        "of the last 24 hours.\n\n"
        "Rules:\n"
        "- Organise by country with clear headings, preceded by a one-paragraph overview "
        "of the biggest story overall.\n"
        "- For each story: what happened, and why it matters — background, implications, "
        "key reactions and stakeholders. For developing stories, note what to watch next.\n"
        "- Sources with links are MANDATORY: after every story, on its own line, add the "
        "source link(s) as Markdown, e.g. \"[BBC News](https://...)\", one per line. If "
        "several fetched items informed the story, list them all. Never present a story "
        "without at least one link.\n"
        "- Translate Swedish, Danish and French headlines and summaries into English.\n"
        "- Depth over breadth: pick the most significant stories and treat them thoroughly; "
        "this is a thorough but digestible morning read, not an exhaustive report.\n"
        "- Guðmundur has a standing interest in forestry, woodland restoration and "
        "conservation. Whenever a story in these areas appears — in any country or in "
        "the Topics feed — include it and give it the same thorough treatment. Do not "
        "force coverage if nothing relevant appears.\n"
        "- The 'Topics' feed (Guardian Environment) is a source for forestry/conservation "
        "and other environmental stories across countries; fold relevant items into the "
        "country sections (or the UK section for UK-only stories) rather than creating "
        "a separate Topics heading.\n"
        "- If a country has no significant news, say so briefly.\n"
        "- Write in UK English (colour, organise, realise).\n"
        "- If a headline's summary is thin, present the story cautiously from the headline "
        "alone rather than inventing detail.\n"
        f"- Date the brief {today}.\n\n"
        "Fetched headlines (last 24 hours):\n\n" + "\n\n".join(sections)
    )

    request = ChatCompletionRequest(
        model=model,
        messages=[UserMessage(content=prompt)],
    )
    result = await mistralai_chat_complete(request)
    content = result.choices[0].message.content
    if isinstance(content, list):
        # Model replies may arrive as content blocks (e.g. thinking + text).
        # Keep ONLY the text of each block; skip thinking content entirely.
        parts = []
        for block in content:
            text = None
            if isinstance(block, dict):
                text = block.get("text")
            else:
                text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        content = "\n\n".join(parts)
    return content or ""


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

        brief = await write_brief(items, data.model)
        return NewsBriefOutput(
            brief_markdown=brief,
            feeds_ok=feeds_ok,
            feeds_failed=feeds_failed,
            item_count=len(items),
            model=data.model,
        )
