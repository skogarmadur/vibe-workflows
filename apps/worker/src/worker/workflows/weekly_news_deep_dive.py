"""Weekly news deep-dive workflow: UK, Ireland, Sweden, Denmark.

Sunday complement to the daily brief: fetches a wider set of domestic-language
RSS feeds, then writes Guðmundur's weekly deep-dive report via a Mistral model
call. Deeper analysis than the daily brief; draws on commentary and analysis
angles where the feeds expose them. Designed for an unattended scheduled run:
reads only public feeds, no user-scope credentials. Deliverable is the
execution result (read in Studio / delivered via the 08:35 relay task).

Activity names are suffixed `_deepdive` because activity names must be unique
across all workflows served by this worker.
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

# Public RSS feeds, verified reachable on 2026-11-12. Wider than the daily brief:
# adds Irish Times, SvD, Expressen, GP, Berlingske and the Guardian's world feed.
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
)

COUNTRIES = ("UK", "Ireland", "Sweden", "Denmark")

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


@workflows.activity()
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


@workflows.activity()
async def write_deep_dive_report(feed_items: list[dict], model: str) -> str:
    """Call a Mistral model to write the weekly deep-dive report."""
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
        f"You are writing Guðmundur's weekly news deep-dive report for **Sunday {today}**, "
        "covering the United Kingdom, Ireland, Sweden and Denmark, reviewing the week "
        "just ended and looking ahead.\n\n"
        "Rules:\n"
        "- Organise by country with clear headings, preceded by a one-paragraph overview "
        "of the week across all four countries.\n"
        "- This is the fullest analysis of the week: deeper and more exhaustive than the "
        "daily morning brief. For each significant story: what happened, why it matters — "
        "background, implications, key reactions and stakeholders — and what to watch next. "
        "Trace developments across the week rather than listing single headlines.\n"
        "- Draw on domestic-language sources (Swedish, Danish, Irish) for local perspective "
        "and commentary; translate into English. Cross-reference international sources "
        "for cross-border stories.\n"
        "- Cite sources with the links given. If a summary is thin, present the story "
        "cautiously from the headline alone rather than inventing detail.\n"
        "- End with a short 'The week ahead' section of things to watch in each country.\n"
        "- If a country had a quiet week, say so briefly.\n"
        "- Write in UK English (colour, organise, realise).\n"
        f"- Date the report Sunday {today}.\n\n"
        "Fetched headlines:\n\n" + "\n\n".join(sections)
    )

    request = ChatCompletionRequest(
        model=model,
        messages=[UserMessage(content=prompt)],
    )
    result = await mistralai_chat_complete(request)
    return result.choices[0].message.content or ""


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

        report = await write_deep_dive_report(items, data.model)
        return DeepDiveOutput(
            report_markdown=report,
            feeds_ok=feeds_ok,
            feeds_failed=feeds_failed,
            item_count=len(items),
            model=data.model,
        )
