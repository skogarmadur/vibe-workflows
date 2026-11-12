"""Weekly news deep-dive workflow: UK, Ireland, Sweden, Denmark.

Fetches headlines from domestic-language RSS feeds for the last 7 days,
then writes Guðmundur's Sunday deep-dive report via a Mistral model call.
Designed for an unattended scheduled run (Sunday 08:30): reads only public
feeds, no user-scope credentials. Deliverable is the execution result,
relayed into chat by a scheduled task at 08:35.
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

# Public RSS feeds, verified reachable on 2026-11-12. One feed per (country, source).
# DR and Aftonbladet publish no working public RSS and are therefore not included.
FEEDS: tuple[tuple[str, str, str], ...] = (
    ("UK", "BBC News", "https://feeds.bbci.co.uk/news/uk/rss.xml"),
    ("UK", "The Guardian", "https://www.theguardian.com/uk/rss"),
    ("Ireland", "RTÉ", "https://www.rte.ie/feeds/rss/?index=/news/"),
    ("Ireland", "TheJournal.ie", "https://www.thejournal.ie/feed/"),
    ("Ireland", "The Irish Times", "https://www.irishtimes.com/arc/outboundfeeds/rss/"),
    ("Ireland", "Irish Independent", "https://www.independent.ie/rss"),
    ("Sweden", "SVT Nyheter", "https://www.svt.se/nyheter/rss.xml"),
    ("Sweden", "Dagens Nyheter", "https://www.dn.se/nyheter/rss/"),
    ("Sweden", "Expressen", "https://feeds.expressen.se/nyheter/"),
    ("Sweden", "Svenska Dagbladet", "https://www.svd.se/?service=rss&type=senastenytt"),
    ("Sweden", "Göteborgs-Posten", "https://www.gp.se/rss"),
    ("Denmark", "Politiken", "https://politiken.dk/rss/senestenyt.rss"),
    ("Denmark", "Ekstra Bladet", "https://ekstrabladet.dk/rssfeed/all/"),
    ("Denmark", "Berlingske", "https://www.berlingske.dk/content/rss"),
    ("Denmark", "B.T.", "https://www.bt.dk/bt/seneste/rss"),
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
    """Input for the weekly deep dive."""

    max_items_per_feed: int = Field(default=15, ge=1, le=40)
    model: str = "mistral-large-latest"
    dry_run: bool = False


class FeedItem(BaseModel):
    country: str
    source: str
    title: str
    link: str
    summary: str


class DeepDiveOutput(BaseModel):
    report_markdown: str
    feeds_ok: list[str]
    feeds_failed: list[str]
    item_count: int
    model: str


@workflows.activity()
async def fetch_feed(country: str, source: str, url: str, max_items: int) -> list[dict]:
    """Fetch and parse one public RSS feed. Failures return [] (handled upstream)."""
    from datetime import datetime, timedelta, timezone
    from email.utils import parsedate_to_datetime

    import httpx

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(15.0),
        headers={"User-Agent": "Mozilla/5.0 (compatible; news-deep-dive-workflow/1.0)"},
        follow_redirects=True,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.text

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)

    items: list[dict] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return items
    for element in root.iter("item"):
        title = _clean(_text(element, "title"))
        link = (_text(element, "link") or "").strip()
        summary = _clean(_text(element, "description"))
        if not (title and link):
            continue
        # Keep items from the last 7 days when a date is parseable; keep
        # undated items (some feeds omit pubDate) so nothing is lost.
        pub_date = _text(element, "pubDate")
        if pub_date:
            try:
                parsed = parsedate_to_datetime(pub_date)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                if parsed < cutoff:
                    continue
            except (TypeError, ValueError):
                pass
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
async def write_report(feed_items: list[dict], model: str) -> str:
    """Call a Mistral model to write the deep-dive report from the headlines."""
    from datetime import datetime

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
        f"You are writing Guðmundur's weekly Sunday news deep-dive report for **{today}**, "
        "covering the United Kingdom, Ireland, Sweden and Denmark, from news of the "
        "last 7 days.\n\n"
        "Rules:\n"
        "- Organise by country with clear headings, preceded by a one-paragraph overview "
        "of the biggest themes of the week across all four countries.\n"
        "- For each significant story: what happened this week, and why it matters — "
        "background, implications, key reactions and stakeholders. Where the story "
        "developed over several days, trace the arc of the week rather than listing "
        "individual headlines.\n"
        "- Note what to watch next week for developing stories.\n"
        "- Cite sources with the links given. Translate Swedish and Danish headlines and "
        "summaries into English.\n"
        "- Depth over breadth: pick the most significant stories and treat them "
        "thoroughly and analytically. This is the fullest weekly analysis — noticeably "
        "longer and more exhaustive than the daily morning brief.\n"
        "- If a country has no significant news, say so briefly.\n"
        "- Write in UK English (colour, organise, realise).\n"
        "- If a headline's summary is thin, present the story cautiously from the "
        "headline alone rather than inventing detail.\n"
        f"- Date the report {today}.\n\n"
        "Fetched headlines (last 7 days):\n\n" + "\n\n".join(sections)
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
            feed_items = await fetch_feed(country, source, url, data.max_items_per_feed)
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

        report = await write_report(items, data.model)
        return DeepDiveOutput(
            report_markdown=report,
            feeds_ok=feeds_ok,
            feeds_failed=feeds_failed,
            item_count=len(items),
            model=data.model,
        )
