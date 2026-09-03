"""Shared fixtures for harness tests.

Every test gets an isolated run directory and a fake HTTP transport so
web_search/web_fetch never hit the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.config import LabConfig
from lab.data_cache import seed_default_mix
from lab.http import HttpResponse
from lab.supervisor import Supervisor


class FakeTransport:
    """DuckDuckGo-shaped JSON body; records requested URLs."""

    def __init__(self, body: bytes | None = None) -> None:
        self.urls: list[str] = []
        self.body = body or (
            b'{"Heading":"TinyStories","AbstractText":"A small-story corpus.",'
            b'"AbstractURL":"https://example.com","RelatedTopics":[]}'
        )

    def get(self, url: str, timeout: float) -> HttpResponse:
        self.urls.append(url)
        return HttpResponse(200, self.body, "application/json")


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport()


@pytest.fixture
def sup(tmp_path: Path, transport: FakeTransport) -> Supervisor:
    cache = tmp_path / "hf_cache"
    seed_default_mix(cache)
    cfg = LabConfig(
        run_dir=tmp_path / "run",
        allow_network=True,
        research_max_tool_calls=20,
        research_max_seconds=60,
        dummy_job_seconds=0.0,
        data_cache_dir=cache,
    )
    return Supervisor(cfg, transport=transport)
