"""Bounded public web-page retrieval and conservative text extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class WebDocument:
    url: str
    title: str
    text: str
    retrieved_at: str
    content_type: str


FetchTransport = Callable[[str, int], tuple[bytes, str]]


class WebReader:
    def __init__(self, fetch: FetchTransport | None = None, max_bytes: int = 1_000_000) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.fetch = fetch or self._fetch
        self.max_bytes = max_bytes

    def read(self, url: str) -> WebDocument:
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError("url must use http:// or https://")
        raw, content_type = self.fetch(url, self.max_bytes)
        if len(raw) > self.max_bytes:
            raise ValueError("response exceeded max_bytes")
        html = raw.decode("utf-8", errors="replace")
        title = self._extract_title(html) or url
        text = self._extract_text(html)
        if not text:
            raise ValueError("web page contained no extractable text")
        from datetime import datetime, timezone

        return WebDocument(url, title, text, datetime.now(timezone.utc).isoformat(), content_type)

    def _fetch(self, url: str, max_bytes: int) -> tuple[bytes, str]:
        request = Request(url, headers={"User-Agent": "AAB-research/1.0"})
        try:
            with urlopen(request, timeout=10) as response:
                content_type = response.headers.get("Content-Type", "application/octet-stream")
                return response.read(max_bytes + 1), content_type
        except Exception as exc:
            raise RuntimeError(f"web retrieval failed: {exc}") from exc

    @staticmethod
    def _extract_title(html: str) -> str:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""

    @staticmethod
    def _extract_text(html: str) -> str:
        without_noise = re.sub(
            r"<(script|style|noscript|template)[^>]*>.*?</\1>",
            " ",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(r"<[^>]+>", " ", without_noise)
        return re.sub(r"\s+", " ", text).strip()
