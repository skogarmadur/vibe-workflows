"""Weekly news deep-dive workflow: UK, Ireland, Sweden, Denmark, France.

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

# Public RSS feeds, verified reachable on 2026-11-12. Wider than the daily brief:
# adds Irish Times, SvD, Expressen, GP, Berlingske, the Guardian's world feed,
# four French outlets, and the Guardian's environment feed for standing interests.
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
    start_to_close_timeout=timedelta(minutes=25),
    retry_policy_max_attempts=2,
)
async def write_deep_dive_report(feed_items: list[dict], model: str) -> str:
    """Write the weekly deep-dive report via several smaller Mistral model calls.

    The platform HTTP client applies a read timeout (~5 minutes) to each
    model call, so one giant call covering all 19 feeds reliably times out.
    Writing one section per country (plus a short overview pass) keeps every
    individual call short enough to complete.
    """
    from datetime import datetime

    # The current date is computed in the activity (allowed: activities may
    # use non-deterministic sources) so the model never has to guess it.
    today = datetime.now().strftime("%A %-d %B %Y")

    grouped: dict[str, list[dict]] = {country: [] for country in COUNTRIES}
    for item in feed_items:
        grouped.setdefault(item["country"], []).append(item)

    def _items_text(country: str) -> str:
        lines = [f"## {country}"]
        for item in grouped[country]:
            summary = f" — {item['summary']}" if item.get("summary") else ""
            lines.append(f"- {item['title']} ({item['source']}){summary}\n  {item['link']}")
        return "\n".join(lines)

    async def _chat(prompt: str) -> str:
        request = ChatCompletionRequest(model=model, messages=[UserMessage(content=prompt)])
        result = await mistralai_chat_complete(request)
        content = result.choices[0].message.content
        if isinstance(content, list):
            # Model replies may arrive as content blocks (thinking + text):
            # keep ONLY the text of each block; skip thinking entirely.
            parts = []
            for block in content:
                text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            content = "\n\n".join(parts)
        return content or ""

    common_rules = (
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
    )

    sections: list[str] = []
    for country in COUNTRIES:
        items = grouped.get(country, [])
        if not items:
            if country == "Topics":
                continue
            sections.append(f"## {country}\n\nNo feeds available this week.")
            continue
        if country == "Topics":
            prompt = (
                f"You are writing the environment section of Guðmundur's weekly news "
                f"deep-dive report for **{today}**, from the Guardian environment feed. "
                "Guðmundur's standing interest is forestry, woodland restoration and "
                "conservation: lead with those stories where present, then other major "
                "environmental news. Give the section the heading "
                "'## Forestry & Conservation' (subsume other environmental stories under "
                "it with a subheading if needed).\n\n"
                "Rules:\n" + common_rules + "\n"
                "Fetched headlines:\n\n" + _items_text(country)
            )
        else:
            prompt = (
                f"You are writing the {country} section of Guðmundur's weekly news "
                f"deep-dive report for **{today}**, reviewing the week just ended and "
                "looking ahead.\n\n"
                "Rules:\n" + common_rules +
                "- Write ONLY this country's section, starting with a '## Country' "
                "heading. No overall overview; that is written separately.\n\n"
                "Fetched headlines:\n\n" + _items_text(country)
            )
        sections.append(await _chat(prompt))

    overview_prompt = (
        f"You are writing the opening overview of Guðmundur's weekly news deep-dive "
        f"report for **{today}** (Sunday), covering the United Kingdom, Ireland, Sweden, "
        "Denmark and France.\n\n"
        "Below are the already-written country sections of the report. Write a single "
        "opening paragraph (no heading) giving the big picture of the week across the "
        "five countries, then a short '## The week ahead' section with one or two things "
        "to watch in each country. Write in UK English. Do not repeat the sections in "
        "detail.\n\n" + "\n\n".join(sections)
    )
    overview = await _chat(overview_prompt)

    return overview + "\n\n" + "\n\n".join(sections)

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
