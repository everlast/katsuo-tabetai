from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_MODEL
from .models import HotelLocation, RestaurantCandidateInput, ScrapedPage


@dataclass
class KatsuoContext:
    hotel: HotelLocation
    max_distance_km: float
    output_dir: Path
    model: str = DEFAULT_MODEL
    trace_id: str = "pending"
    collected_candidates: list[RestaurantCandidateInput] = field(default_factory=list)
    pending_candidates: list[RestaurantCandidateInput] = field(default_factory=list)
    scraped_pages: dict[str, ScrapedPage] = field(default_factory=dict)
    candidate_rejections: list[str] = field(default_factory=list)
    cached_candidates_loaded: int = 0
    cached_candidates_written: int = 0
    candidate_save_calls: int = 0
    candidates_saved: bool = False
    evaluation_tool_calls: int = 0
    handoff_calls: int = 0
    scrape_calls: int = 0
    progress_callback: Callable[[str], None] | None = None

    @property
    def candidates_path(self) -> Path:
        return self.output_dir / "restaurant_candidates.json"

    @property
    def restaurant_cache_dir(self) -> Path:
        return self.output_dir / "restaurants"

    @property
    def discovered_candidates_path(self) -> Path:
        return self.output_dir / "discovered_restaurants.json"

    @property
    def top_five_path(self) -> Path:
        return self.output_dir / "top5.json"

    @property
    def html_path(self) -> Path:
        return self.output_dir / "top5.html"

    @property
    def context_markdown_path(self) -> Path:
        return self.output_dir / "context.md"

    @property
    def scrape_manifest_path(self) -> Path:
        return self.output_dir / "scrape_manifest.json"

    @property
    def run_manifest_path(self) -> Path:
        return self.output_dir / "run_manifest.json"
