from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import json


@dataclass
class HttpResponse:
    status: int
    body: bytes
    content_type: str = ""


class Transport(Protocol):
    def get(self, url: str, timeout: float) -> HttpResponse: ...


class UrllibTransport:
    def get(self, url: str, timeout: float) -> HttpResponse:
        req = Request(url, headers={"User-Agent": "lab-harness/0.1"})
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — scheme checked by caller
            return HttpResponse(
                status=getattr(resp, "status", 200),
                body=resp.read(),
                content_type=resp.headers.get("Content-Type", ""),
            )


def assert_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("only http/https URLs are allowed")
    if not parsed.netloc:
        raise ValueError("url host is required")


def duckduckgo_search_url(query: str) -> str:
    return "https://api.duckduckgo.com/?" + urlencode(
        {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    )


def parse_duckduckgo(body: bytes) -> list[dict[str, str]]:
    data = json.loads(body.decode("utf-8", errors="replace"))
    hits: list[dict[str, str]] = []
    abstract = (data.get("AbstractText") or "").strip()
    abstract_url = (data.get("AbstractURL") or "").strip()
    if abstract:
        hits.append({"title": data.get("Heading") or query_fallback(data), "url": abstract_url, "snippet": abstract})
    for item in data.get("RelatedTopics") or []:
        if not isinstance(item, dict):
            continue
        if "Topics" in item:
            continue
        text = (item.get("Text") or "").strip()
        url = (item.get("FirstURL") or "").strip()
        if text:
            hits.append({"title": text[:80], "url": url, "snippet": text})
        if len(hits) >= 8:
            break
    return hits[:8]


def query_fallback(data: dict) -> str:
    return str(data.get("Heading") or "result")
