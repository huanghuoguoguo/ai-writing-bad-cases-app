"""CLI integration tests."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ai_badcase_app.cli", *args],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )


def _make_input_file(text: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(text)
    tmp.close()
    return Path(tmp.name)


def test_cli_text_output():
    path = _make_input_file("不绕弯子，直接说重点。\n\n完全普通的一段话。")
    result = _run_cli("--input", str(path))
    assert result.returncode == 0
    assert "score=" in result.stdout
    assert "不绕弯子类开场" in result.stdout
    assert "dimensions:" in result.stdout
    path.unlink()


def test_cli_json_output():
    path = _make_input_file("真正重要的不是速度，而是你是否能长期坚持。")
    result = _run_cli("--input", str(path), "--format", "json")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "score" in data[0]
    assert "hits" in data[0]
    assert "diagnostic_dimensions" in data[0]["hits"][0]
    path.unlink()


def test_cli_genre_filter():
    path = _make_input_file("稳稳接住你。")
    result = _run_cli("--input", str(path), "--genre", "narrative", "--format", "json")
    assert result.returncode == 0
    data = json.loads(result.stdout)
    # steadily_catch_you is cross-genre (argumentative + narrative), so it should still match
    assert len(data) >= 1
    path.unlink()


def test_cli_no_input_file():
    result = _run_cli("--input", "/nonexistent/file.txt")
    assert result.returncode != 0
