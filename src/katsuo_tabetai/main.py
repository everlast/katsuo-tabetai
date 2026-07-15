from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from agents import AgentsException
from dotenv import find_dotenv, load_dotenv

from .context import KatsuoContext
from .models import HotelLocation
from .workflow import (
    InsufficientResearchCandidatesError,
    InvalidResearchOutputError,
    NoValidResearchCandidatesError,
    run_katsuo_workflow,
)

DEFAULT_HOTEL_NAME = (
    "ザ クラウンパレス高知（2026年8月1日からANAクラウンプラザホテル高知 by IHG）"
)
DEFAULT_HOTEL_LATITUDE = 33.5577702
DEFAULT_HOTEL_LONGITUDE = 133.5339508


def load_project_environment() -> bool:
    """Load the nearest project .env without replacing shell-provided values."""
    dotenv_path = find_dotenv(usecwd=True)
    return bool(dotenv_path) and load_dotenv(dotenv_path, override=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a sourced katsuo restaurant TOP 5 near a hotel in Kochi."
    )
    parser.add_argument("--hotel-name", default=DEFAULT_HOTEL_NAME)
    parser.add_argument("--hotel-lat", type=float, default=DEFAULT_HOTEL_LATITUDE)
    parser.add_argument("--hotel-lon", type=float, default=DEFAULT_HOTEL_LONGITUDE)
    parser.add_argument("--max-distance-km", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override. By default the Agents SDK setting is used.",
    )
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument(
        "--research-attempts",
        type=int,
        default=3,
        help=(
            "Maximum Web research attempts before failing on insufficient "
            "validated in-range candidates."
        ),
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is required for WebSearchTool and tracing. "
            "Create .env from .env.example and set the key."
        )
    if args.max_distance_km <= 0:
        raise SystemExit("--max-distance-km must be greater than zero.")
    if args.research_attempts <= 0:
        raise SystemExit("--research-attempts must be greater than zero.")

    context = KatsuoContext(
        hotel=HotelLocation(
            name=args.hotel_name,
            latitude=args.hotel_lat,
            longitude=args.hotel_lon,
        ),
        max_distance_km=args.max_distance_km,
        output_dir=args.output_dir.resolve(),
    )
    outcome = await run_katsuo_workflow(
        context=context,
        model=args.model,
        max_turns=args.max_turns,
        research_attempts=args.research_attempts,
    )
    print(
        json.dumps(
            {
                "last_agent": outcome.last_agent,
                "trace_id": outcome.trace_id,
                "trace_dashboard": "https://platform.openai.com/traces",
                "candidates_json": str(context.candidates_path),
                "top_five_json": str(context.top_five_path),
                "html": str(context.html_path),
                "rejected_research_candidates": context.candidate_rejections,
                "audit": outcome.audit.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> None:
    load_project_environment()
    try:
        exit_code = asyncio.run(_run(build_parser().parse_args()))
    except (
        AgentsException,
        InsufficientResearchCandidatesError,
        InvalidResearchOutputError,
        NoValidResearchCandidatesError,
    ) as exc:
        raise SystemExit(f"Katsuo workflow failed: {exc}") from None
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
