from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import HotelLocation


@dataclass
class KatsuoContext:
    hotel: HotelLocation
    max_distance_km: float
    output_dir: Path
    candidate_save_calls: int = 0
    candidates_saved: bool = False
    evaluation_tool_calls: int = 0
    handoff_calls: int = 0
    handoff_summary: str | None = None

    @property
    def candidates_path(self) -> Path:
        return self.output_dir / "restaurant_candidates.json"

    @property
    def top_five_path(self) -> Path:
        return self.output_dir / "top5.json"

    @property
    def html_path(self) -> Path:
        return self.output_dir / "top5.html"
