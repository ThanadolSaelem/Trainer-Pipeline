"""Every page must render without exceptions, both on an empty workspace and with runs present."""
import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["home", "data", "perplexity", "train", "evaluate", "compare", "jobs", "settings"]


def _run(page: str):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "pages" / f"{page}.py"), default_timeout=60)
    at.run()
    return at


@pytest.fixture()
def empty_workspace(tmp_path, monkeypatch):
    cfg = tmp_path / "panel.yaml"
    cfg.write_text(yaml.safe_dump({"runs_dir": str(tmp_path / "runs")}))
    monkeypatch.setenv("LLM_PANEL_SETTINGS", str(cfg))
    monkeypatch.chdir(ROOT)
    for m in [m for m in sys.modules if m == "core" or m.startswith("core.")]:
        del sys.modules[m]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_on_empty_workspace(empty_workspace, page):
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.skipif(not os.environ.get("LLM_PANEL_E2E_SETTINGS"), reason="set LLM_PANEL_E2E_SETTINGS to a workspace with runs")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_runs(monkeypatch, page):
    monkeypatch.setenv("LLM_PANEL_SETTINGS", os.environ["LLM_PANEL_E2E_SETTINGS"])
    monkeypatch.chdir(ROOT)
    for m in [m for m in sys.modules if m == "core" or m.startswith("core.")]:
        del sys.modules[m]
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]
