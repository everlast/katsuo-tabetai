from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from agents import AgentsException, set_default_openai_client
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI

from .config import (
    DEFAULT_DISCOVERY_ATTEMPTS,
    DEFAULT_MODEL,
    DEFAULT_REVIEW_ENRICHMENT_ATTEMPTS,
)
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
        default=DEFAULT_MODEL,
        help=f"OpenAI model ID. Defaults to the required model {DEFAULT_MODEL}.",
    )
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument(
        "--api-timeout-seconds",
        type=float,
        default=300.0,
        help="Maximum wait for one OpenAI API request. Defaults to 300 seconds.",
    )
    parser.add_argument(
        "--api-max-retries",
        type=int,
        default=0,
        help="Automatic OpenAI API retries after a failed request. Defaults to 0.",
    )
    parser.add_argument(
        "--workflow-timeout-seconds",
        type=float,
        default=10800.0,
        help="Maximum duration of the complete workflow. Defaults to 10800 seconds.",
    )
    parser.add_argument(
        "--discovery-attempts",
        type=int,
        default=DEFAULT_DISCOVERY_ATTEMPTS,
        help=(
            "Number of restaurant discovery attempts. "
            f"Defaults to {DEFAULT_DISCOVERY_ATTEMPTS}."
        ),
    )
    parser.add_argument(
        "--review-enrichment-attempts",
        type=int,
        default=DEFAULT_REVIEW_ENRICHMENT_ATTEMPTS,
        help=(
            "Number of review enrichment attempts. "
            f"Defaults to {DEFAULT_REVIEW_ENRICHMENT_ATTEMPTS}."
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
    if args.discovery_attempts < 0:
        raise SystemExit("--discovery-attempts must be zero or greater.")
    if args.review_enrichment_attempts < 0:
        raise SystemExit("--review-enrichment-attempts must be zero or greater.")
    if args.discovery_attempts + args.review_enrichment_attempts <= 0:
        raise SystemExit(
            "At least one discovery or review enrichment attempt is required."
        )
    if args.api_timeout_seconds <= 0:
        raise SystemExit("--api-timeout-seconds must be greater than zero.")
    if args.api_max_retries < 0:
        raise SystemExit("--api-max-retries must be zero or greater.")
    if args.workflow_timeout_seconds <= 0:
        raise SystemExit("--workflow-timeout-seconds must be greater than zero.")

    context = KatsuoContext(
        hotel=HotelLocation(
            name=args.hotel_name,
            latitude=args.hotel_lat,
            longitude=args.hotel_lon,
        ),
        max_distance_km=args.max_distance_km,
        output_dir=args.output_dir.resolve(),
        model=args.model,
        progress_callback=lambda message: print(
            f"[katsuo] {message}",
            file=sys.stderr,
            flush=True,
        ),
    )
    client = AsyncOpenAI(
        timeout=args.api_timeout_seconds,
        max_retries=args.api_max_retries,
    )
    set_default_openai_client(client)
    try:
        async with asyncio.timeout(args.workflow_timeout_seconds):
            outcome = await run_katsuo_workflow(
                context=context,
                model=args.model,
                max_turns=args.max_turns,
                discovery_attempts=args.discovery_attempts,
                review_enrichment_attempts=args.review_enrichment_attempts,
            )
    finally:
        await client.close()
    print(
        json.dumps(
            {
                "last_agent": outcome.last_agent,
                "trace_id": outcome.trace_id,
                "trace_dashboard": "https://platform.openai.com/traces",
                "model": outcome.model,
                "restaurant_cache_dir": str(context.restaurant_cache_dir),
                "discovered_restaurants": str(context.discovered_candidates_path),
                "collected_restaurants": len(context.collected_candidates),
                "evaluation_eligible_restaurants": len(context.pending_candidates),
                "context_markdown": str(context.context_markdown_path),
                "scrape_manifest": str(context.scrape_manifest_path),
                "run_manifest": str(context.run_manifest_path),
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
    args = build_parser().parse_args()
    try:
        exit_code = asyncio.run(_run(args))
    except KeyboardInterrupt:
        raise SystemExit("Katsuo workflow interrupted.") from None
    except TimeoutError:
        raise SystemExit(
            "Katsuo workflow timed out after "
            f"{args.workflow_timeout_seconds:g} seconds."
        ) from None
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
