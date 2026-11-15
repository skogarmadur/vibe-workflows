"""Weekly news deep-dive workflow: UK, Ireland, Sweden, Denmark, France.

Sunday complement to the daily brief: fetches a wider set of domestic-language
RSS feeds, then writes Guðmundur's weekly deep-dive report via several small
Mistral model calls (one per country, plus a short overview pass). The
platform HTTP client read-times-out long model calls (~5 minutes), so no
single call may cover every feed at once, and each country is its own
activity so a retry only replays one small call.

Activity names are suffixed `_deepdive` because activity names must be unique
across all workflows served by this worker.
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

FEEDS: tuple[tuple[str, str, str], ...] = (
    ("UK", "BBC News", "https://feeds.bbci.co.uk/news/uk/rss.xml"),
    ("UK", "The Guardian", "https://www.theguardian.com/uk/rss"),
    ("UK", "The Guardian (World)", "https://www.theguardian.com/world/rss"),
    ("Ireland", "RTÉ", "https://www.rte.ie/feeds/rss/?index=/news/"),
    ("Ireland", "The Irish Times", "https://www.irishtimes.com/arc/outboundfeeds/rss/"),
    ("Ireland", "TheJournal.ie", "https://www.thejournal.ie/feed/"),
    ("Sweden", "SVT Nyheter", "https://www.svt.se/nyheter/rss.xml"),
    ("Sweden", "Dagens Nyheter", "https://www.dn.se/nyheter/rss/"),
    ("Sweden", "Svenska Dagbladet", "https://www.svd.se/?service=rss"),
    ("Sweden", "Expressen", "https://www.expressen.se/rss/nyheter/"),
    ("Sweden", "Göteborgs-Posten", "https://www.gp.se/rss"),
    ("Denmark", "Politiken", "https://politiken.dk/rss/senestenyt.rss"),
    ("Denmark", "Ekstra Bladet", "https://ekstrabladet.dk/rssfeed/all/"),
    ("Denmark", "Berlingske", "https://www.berlingske.dk/content/rss"),
    ("France", "Le Monde", "https://www.lemonde.fr/rss/une.xml"),
    ("France", "Le Figaro", "https://www.lefigaro.fr/rss/figaro_actualites.xml"),
    ("France", "franceinfo", "https://www.francetvinfo.fr/titres.rss"),
    ("France", "Ouest-France", "https://www.ouest-france.fr/rss-en-continu.xml"),
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
    for descendant in element.iter(tag):
        if descendant.text:
            return descendant.text
    return None


def _extract_text(content) -> str:
    if isinstance(content, list):
        parts = []
        for block in content:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "\n\n".join(parts)
    return content or ""


class DeepDiveInput(BaseModel):
    """Input for the weekly news deep dive."""

    max_items_per_feed: int = Field(default=15, ge=1, le=40)
    model: str = "mistral-large-latest"
    dry_run: bool = False


class DeepDiveOutput(BaseModel):
    report_markdown: str
    feeds_ok: list[str]
    feeds_failed: list[str]
    item_count: int
    model: str


@workflows.activity(
    start_to_close_timeout=timedelta(seconds=90),
    retry_policy_max_attempts=2,
)
async def fetch_feed_deepdive(country: str, source: str, url: str, max_items: int) -> list[dict]:
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
async def write_country_section_deepdive(country: str, feed_items: list[dict], model: str) -> dict:
    """Write one country's section via a single small model call.

    Kept small deliberately: the platform HTTP client read-times-out long
    model calls, and each country is its own activity so a retry only replays
    one small call. Returns the section plus machine-readable highlights.
    """
    from datetime import datetime

    today = datetime.now().strftime("%A %-d %B %Y")

    lines = [f"## {country}"]
    for item in feed_items:
        summary = f" — {item['summary']}" if item.get("summary") else ""
        lines.append(f"- {item['title']} ({item['source']}){summary}\n  {item['link']}")
    items_text = "\n".join(lines)

    if country == "Topics":
        intro = (
            "You are writing the environment section of Guðmundur's weekly news "
            f"deep-dive report for **{today}**, from the Guardian environment feed. "
            "Guðmundur's standing interest is forestry, woodland restoration and "
            "conservation: lead with those stories where present, then other major "
            "environmental news. Give the section the heading "
            "'## Forestry & Conservation' (subsume other environmental stories under "
            "it with a subheading if needed).\n\n"
        )
    else:
        intro = (
            f"You are writing the {country} section of Guðmundur's weekly news "
            f"deep-dive report for **{today}**, reviewing the week just ended and "
            "looking ahead.\n\n"
        )

    prompt = (
        intro
        + "Rules:\n"
        "- This is the fullest analysis of the week: deeper and more exhaustive than the "
        "daily morning brief. For each significant story: what happened, why it matters — "
        "background, implications, key reactions and stakeholders — and what to watch next. "
        "Trace developments across the week rather than listing single headlines.\n"
        "- Draw on domestic-language sources for local perspective and commentary; "
        "translate into English.\n"
        "- Guðmundur has a standing interest in forestry, woodland restoration and "
        "conservation. Whenever a story in these areas appears, include it with the same "
        "thorough treatment. Do not force coverage if nothing relevant appears.\n"
        "- Sources with links are MANDATORY: after every story, on its own line, add the "
        "source link(s) as Markdown, e.g. \"[Politiken](https://...)\". If several fetched "
        "items informed the story, list them all. Never present a story without at least "
        "one link.\n"
        "- If a summary is thin, present the story cautiously from the headline alone "
        "rather than inventing detail.\n"
        "- If the country had a quiet week, say so briefly.\n"
        "- Write in UK English (colour, organise, realise).\n"
        "- End your reply with exactly this machine-readable block (nothing after it):\n"
        "<<<HIGHLIGHTS>>>\n"
        "- <one-line summary of the week's biggest story>\n"
        "- <one-line summary of the second biggest story>\n"
        "- <one-line summary of the third biggest story>\n"
        "<<<WATCH>>>\n"
        "- <one or two things to watch in this country next week>\n\n"
        "Fetched headlines:\n\n" + items_text
    )

    request = ChatCompletionRequest(model=model, messages=[UserMessage(content=prompt)])
    result = await mistralai_chat_complete(request)
    content = _extract_text(result.choices[0].message.content)

    highlights: list[str] = []
    watch: list[str] = []
    section = content
    if "<<<HIGHLIGHTS>>>" in content:
        section, rest = content.split("<<<HIGHLIGHTS>>>", 1)
        hl_block, watch_block = rest, ""
        if "<<<WATCH>>>" in rest:
            hl_block, watch_block = rest.split("<<<WATCH>>>", 1)
        highlights = [l.strip()[2:].strip() for l in hl_block.strip().splitlines() if l.strip().startswith("-")]
        watch = [l.strip()[2:].strip() for l in watch_block.strip().splitlines() if l.strip().startswith("-")]
    return {
        "country": country,
        "section_markdown": section.strip(),
        "week_highlights": highlights,
        "watch_next": watch,
    }


@workflows.activity(
    start_to_close_timeout=timedelta(minutes=6),
    retry_policy_max_attempts=3,
)
async def write_overview_deepdive(highlights_text: str, model: str) -> str:
    """Write the opening overview paragraph from the per-country highlights."""
    from datetime import datetime

    today = datetime.now().strftime("%A %-d %B %Y")
    prompt = (
        f"You are writing the opening overview paragraph of Guðmundur's weekly news "
        f"deep-dive report for **{today}** (Sunday), covering the United Kingdom, "
        "Ireland, Sweden, Denmark and France.\n\n"
        "Below are the key points of the week from each country section of the report. "
        "Write a single opening paragraph (no heading, no bullet points) giving the "
        "big picture of the week across the five countries. Write in UK English.\n\n"
        + highlights_text
    )
    request = ChatCompletionRequest(model=model, messages=[UserMessage(content=prompt)])
    result = await mistralai_chat_complete(request)
    return _extract_text(result.choices[0].message.content).strip()


@workflows.workflow.define(name="weekly_news_deep_dive")
class WeeklyNewsDeepDiveWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, data: DeepDiveInput) -> DeepDiveOutput:
        items: list[dict] = []
        feeds_ok: list[str] = []
        feeds_failed: list[str] = []
        for country, source, url in FEEDS:
            feed_items = await fetch_feed_deepdive(country, source, url, data.max_items_per_feed)
            if feed_items:
                items.extend(feed_items)
                feeds_ok.append(f"{source} ({country})")
            else:
                feeds_failed.append(f"{source} ({country})")

        if data.dry_run:
            headlines = "\n".join(f"- {i['title']} — {i['link']}" for i in items)
            return DeepDiveOutput(
                report_markdown=(
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

        sections: list[dict] = []
        for country in COUNTRIES:
            feed_items = grouped.get(country, [])
            if not feed_items:
                if country == "Topics":
                    continue
                sections.append(
                    {
                        "country": country,
                        "section_markdown": f"## {country}\n\nNo feeds available this week.",
                        "week_highlights": [],
                        "watch_next": [],
                    }
                )
                continue
            sections.append(await write_country_section_deepdive(country, feed_items, data.model))

        highlights_text = "\n\n".join(
            f"{s['country']}:\n" + "\n".join(f"- {h}" for h in s["week_highlights"])
            for s in sections
            if s["week_highlights"]
        )
        overview = ""
        if highlights_text:
            overview = await write_overview_deepdive(highlights_text, data.model)

        week_ahead_lines = ["## The week ahead"]
        for s in sections:
            for w in s["watch_next"]:
                week_ahead_lines.append(f"- **{s['country']}**: {w}")
        week_ahead = "\n".join(week_ahead_lines) if len(week_ahead_lines) > 1 else ""

        parts = [p for p in (overview, "\n\n".join(s["section_markdown"] for s in sections), week_ahead) if p]
        report = "\n\n".join(parts)
        return DeepDiveOutput(
            report_markdown=report,
            feeds_ok=feeds_ok,
            feeds_failed=feeds_failed,
            item_count=len(items),
            model=data.model,
        )
