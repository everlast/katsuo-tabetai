from __future__ import annotations

import asyncio
import json

import pytest
from agents.tool_context import ToolContext

from katsuo_tabetai.context import KatsuoContext
from katsuo_tabetai.models import HotelLocation
from katsuo_tabetai.scraping import (
    _ensure_public_http_url,
    _extract_readable_text,
    canonical_url,
    scrape_reference_page,
)

from helpers import _page


def test_readable_text_removes_non_content_elements() -> None:
    title, content = _extract_readable_text(
        """
        <html><head><title>Restaurant</title><style>.hidden{}</style></head>
        <body><script>bad()</script><main><h1>鰹の店</h1><p>塩たたき</p></main></body></html>
        """
    )

    assert title == "Restaurant"
    assert content == "鰹の店\n塩たたき"
    assert "bad" not in content


def test_local_network_urls_are_rejected() -> None:
    with pytest.raises(ValueError, match="Local network"):
        asyncio.run(_ensure_public_http_url("http://localhost/private"))


def test_scrape_tool_records_successful_page(tmp_path, monkeypatch) -> None:
    context = KatsuoContext(
        hotel=HotelLocation(name="Hotel", latitude=33.5, longitude=133.5),
        max_distance_km=2.5,
        output_dir=tmp_path,
    )
    page = _page(
        "https://restaurant.example/menu",
        "Restaurant",
        "Restaurant\nKochi 1\nKatsuo dish\nEnough readable source content.",
    )

    async def fake_fetch(url: str):
        assert url == "https://restaurant.example/menu"
        return page

    monkeypatch.setattr("katsuo_tabetai.scraping.fetch_scraped_page", fake_fetch)
    tool_context = ToolContext(
        context=context,
        tool_name="scrape_reference_page",
        tool_call_id="scrape-call",
        tool_arguments='{"url":"https://restaurant.example/menu"}',
    )

    raw_result = asyncio.run(
        scrape_reference_page.on_invoke_tool(
            tool_context,
            '{"url":"https://restaurant.example/menu"}',
        )
    )
    result = json.loads(raw_result)

    assert result["status"] == "fetched"
    assert result["content_sha256"] == page.content_sha256
    assert context.scrape_calls == 1
    assert canonical_url(page.requested_url) in context.scraped_pages
