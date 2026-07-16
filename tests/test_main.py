from __future__ import annotations

import os
import sys

import pytest
from agents import UserError
from openai import InternalServerError

import katsuo_tabetai
from katsuo_tabetai import main as main_module
from katsuo_tabetai.config import DEFAULT_MODEL
from katsuo_tabetai.main import build_parser, load_project_environment
from katsuo_tabetai.workflow import (
    InsufficientResearchCandidatesError,
    InvalidResearchOutputError,
    NoValidResearchCandidatesError,
)


def test_parser_uses_crown_palais_kochi_as_default_hotel() -> None:
    args = build_parser().parse_args([])

    assert args.hotel_name == (
        "ザ クラウンパレス高知（2026年8月1日からANAクラウンプラザホテル高知 by IHG）"
    )
    assert args.hotel_lat == 33.5577702
    assert args.hotel_lon == 133.5339508
    assert args.max_distance_km == 5.0
    assert args.discovery_attempts == 3
    assert args.review_enrichment_attempts == 27
    assert args.api_timeout_seconds == 300.0
    assert args.api_max_retries == 5
    assert args.workflow_timeout_seconds == 10800.0
    assert args.model == DEFAULT_MODEL == "gpt-5.6-luna"


def test_package_version_attribute_matches_installed_metadata() -> None:
    from importlib.metadata import version

    assert katsuo_tabetai.__version__ == version("katsuo-tabetai")


def test_load_project_environment_reads_dotenv(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_project_environment()

    assert loaded is True
    assert os.environ["OPENAI_API_KEY"] == "from-dotenv"


def test_load_project_environment_preserves_existing_value(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")

    load_project_environment()

    assert os.environ["OPENAI_API_KEY"] == "from-shell"


def test_run_applies_timeouts_and_closes_openai_client(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def close(self) -> None:
            captured["closed"] = True

    async def slow_workflow(**kwargs):
        await main_module.asyncio.sleep(1)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(main_module, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(
        main_module,
        "set_default_openai_client",
        lambda client: captured.update({"default_client": client}),
    )
    monkeypatch.setattr(main_module, "run_katsuo_workflow", slow_workflow)
    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--api-timeout-seconds",
            "12",
            "--api-max-retries",
            "1",
            "--workflow-timeout-seconds",
            "0.01",
        ]
    )

    with pytest.raises(TimeoutError):
        main_module.asyncio.run(main_module._run(args))

    assert captured["timeout"] == 12
    assert captured["max_retries"] == 1
    assert captured["closed"] is True


def test_run_passes_separate_research_attempt_counts(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def close(self) -> None:
            captured["closed"] = True

    async def capture_workflow(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("captured")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(main_module, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(main_module, "set_default_openai_client", lambda client: None)
    monkeypatch.setattr(main_module, "run_katsuo_workflow", capture_workflow)
    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--discovery-attempts",
            "5",
            "--review-enrichment-attempts",
            "12",
        ]
    )

    with pytest.raises(RuntimeError, match="captured"):
        main_module.asyncio.run(main_module._run(args))

    assert captured["discovery_attempts"] == 5
    assert captured["review_enrichment_attempts"] == 12
    assert captured["closed"] is True


def test_main_reports_agents_sdk_errors_without_traceback(monkeypatch) -> None:
    async def fail_run(args):
        raise UserError("candidate validation failed")

    monkeypatch.setattr(main_module, "load_project_environment", lambda: False)
    monkeypatch.setattr(main_module, "_run", fail_run)
    monkeypatch.setattr(sys, "argv", ["katsuo-tabetai"])

    with pytest.raises(
        SystemExit,
        match="Katsuo workflow failed: candidate validation failed",
    ):
        main_module.main()


def test_main_reports_openai_errors_without_traceback(monkeypatch) -> None:
    async def fail_run(args):
        raise InternalServerError(
            "upstream connection terminated",
            response=type(
                "Response",
                (),
                {"request": None, "status_code": 500, "headers": {}},
            )(),
            body=None,
        )

    monkeypatch.setattr(main_module, "load_project_environment", lambda: False)
    monkeypatch.setattr(main_module, "_run", fail_run)
    monkeypatch.setattr(sys, "argv", ["katsuo-tabetai"])

    with pytest.raises(
        SystemExit,
        match="Katsuo workflow failed: upstream connection terminated",
    ):
        main_module.main()


def test_main_reports_no_valid_candidates_without_traceback(monkeypatch) -> None:
    async def fail_run(args):
        raise NoValidResearchCandidatesError("no valid candidates")

    monkeypatch.setattr(main_module, "load_project_environment", lambda: False)
    monkeypatch.setattr(main_module, "_run", fail_run)
    monkeypatch.setattr(sys, "argv", ["katsuo-tabetai"])

    with pytest.raises(
        SystemExit,
        match="Katsuo workflow failed: no valid candidates",
    ):
        main_module.main()


def test_main_reports_insufficient_candidates_without_traceback(monkeypatch) -> None:
    async def fail_run(args):
        raise InsufficientResearchCandidatesError("insufficient candidates")

    monkeypatch.setattr(main_module, "load_project_environment", lambda: False)
    monkeypatch.setattr(main_module, "_run", fail_run)
    monkeypatch.setattr(sys, "argv", ["katsuo-tabetai"])

    with pytest.raises(
        SystemExit,
        match="Katsuo workflow failed: insufficient candidates",
    ):
        main_module.main()


def test_main_reports_invalid_research_output_without_traceback(monkeypatch) -> None:
    async def fail_run(args):
        raise InvalidResearchOutputError("malformed structured JSON")

    monkeypatch.setattr(main_module, "load_project_environment", lambda: False)
    monkeypatch.setattr(main_module, "_run", fail_run)
    monkeypatch.setattr(sys, "argv", ["katsuo-tabetai"])

    with pytest.raises(
        SystemExit,
        match="Katsuo workflow failed: malformed structured JSON",
    ):
        main_module.main()


def test_main_reports_workflow_timeout_without_traceback(monkeypatch) -> None:
    async def fail_run(args):
        raise TimeoutError

    monkeypatch.setattr(main_module, "load_project_environment", lambda: False)
    monkeypatch.setattr(main_module, "_run", fail_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["katsuo-tabetai", "--workflow-timeout-seconds", "42"],
    )

    with pytest.raises(
        SystemExit,
        match="Katsuo workflow timed out after 42 seconds",
    ):
        main_module.main()


def test_main_reports_keyboard_interrupt_without_traceback(monkeypatch) -> None:
    async def interrupt_run(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(main_module, "load_project_environment", lambda: False)
    monkeypatch.setattr(main_module, "_run", interrupt_run)
    monkeypatch.setattr(sys, "argv", ["katsuo-tabetai"])

    with pytest.raises(SystemExit, match="Katsuo workflow interrupted"):
        main_module.main()
