"""Provider-independent public search with an offline injectable transport."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "unknown"


SearchTransport = Callable[[str, int], list[SearchResult]]


class PublicSearch:
    """Search approved public sources without embedding credentials."""

    def __init__(self, transport: SearchTransport | None = None, endpoint: str | None = None) -> None:
        self.transport = transport or self._configured_transport
        self.endpoint = endpoint or os.getenv("AAB_SEARCH_ENDPOINT")

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError("limit must be an integer between 1 and 20")
        return list(self.transport(query.strip(), limit))

    def _configured_transport(self, query: str, limit: int) -> list[SearchResult]:
        if not self.endpoint:
            raise RuntimeError(
                "AAB_SEARCH_ENDPOINT is not configured; inject a transport for offline use"
            )
        request_url = f"{self.endpoint}?{urlencode({'q': query, 'limit': limit})}"
        request = Request(request_url, headers={"User-Agent": "AAB-research/1.0"})
        try:
            with urlopen(request, timeout=10) as response:
                payload = json.load(response)
        except Exception as exc:
            raise RuntimeError(f"public search request failed: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError("search provider response must be a list")
        return [
            SearchResult(
                title=item["title"],
                url=item["url"],
                snippet=item.get("snippet", ""),
                source=item.get("source", "unknown"),
            )
            for item in payload
            if isinstance(item, dict) and isinstance(item.get("title"), str) and isinstance(item.get("url"), str)
        ]
