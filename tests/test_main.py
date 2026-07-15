from __future__ import annotations

import os

from katsuo_tabetai.main import build_parser, load_project_environment


def test_parser_uses_crown_palais_kochi_as_default_hotel() -> None:
    args = build_parser().parse_args([])

    assert args.hotel_name == (
        "ザ クラウンパレス高知（2026年8月1日からANAクラウンプラザホテル高知 by IHG）"
    )
    assert args.hotel_lat == 33.5577702
    assert args.hotel_lon == 133.5339508


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
