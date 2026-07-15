from __future__ import annotations

import os

from katsuo_tabetai.main import load_project_environment


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
