"""Daily news brief workflow: UK, Ireland, Sweden, Denmark, France.

Fetches headlines from domestic-language RSS feeds for the last 24 hours,
then writes Guðmundur's morning brief via several small Mistral model calls
(one per country, plus a short overview pass). The platform HTTP client
read-times-out model calls that take too long (~5 minutes), so no single
call may cover every feed at once, and each country is its own activity so
a retry only replays one small call.

Designed for an unattended scheduled run: reads only public feeds, no
user-scope credentials. Deliverable is the execution result (read in Studio).
"""

from __future__ import annotations

from datetime import timedelta
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

COUNTRIES = ("UK", "Ireland", "Sweden", "Denmark", "France")

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
            "country or are otherwise significant, and include forestry, "
            "woodland restoration and conservation stories wherever they "
            "appear):\n\n" + "\n".join(tlines)
        )

    prompt = (
        f"You are writing the {country} section of Guðmundur's daily morning news "
        f"brief for **{today}**, from news of the last 24 hours.\n\n"
        "Rules:\n"
        "- For each story: what happened, and why it matters — background, implications, "
        "key reactions and stakeholders. For developing stories, note what to watch next.\n"
        "- Sources with links are MANDATORY: after every story, on its own line, add the "
        "source link(s) as Markdown, e.g. \"[BBC News](https://...)\". If several fetched "
        "items informed the story, list them all. Never present a story without at least "
        "one link.\n"
        "- Translate Swedish, Danish and French headlines and summaries into English.\n"
        "- Depth over breadth: pick the most significant stories and treat them thoroughly; "
        "this is a thorough but digestible morning read, not an exhaustive report.\n"
        "- Guðmundur has a standing interest in forestry, woodland restoration and "
        "conservation. Whenever a story in these areas appears, include it and give it "
        "the same thorough treatment. Do not force coverage if nothing relevant appears.\n"
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

        top_stories_text = "\n\n".join(
            f"{s['country']}:\n" + "\n".join(f"- {t}" for t in s["top_stories"])
            for s in sections
            if s["top_stories"]
        )
        overview = ""
        if top_stories_text:
            overview = await write_daily_overview(top_stories_text, data.model)

        parts = [p for p in (overview, "\n\n".join(s["section_markdown"] for s in sections)) if p]
        brief = "\n\n".join(parts)
        return NewsBriefOutput(
            brief_markdown=brief,
            feeds_ok=feeds_ok,
            feeds_failed=feeds_failed,
            item_count=len(items),
            model=data.model,
        )
