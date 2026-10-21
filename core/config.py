"""Workspace settings (panel.yaml) and experiment files (experiments/*.yaml)."""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PANEL_ROOT = Path(__file__).resolve().parents[1]
# LLM_PANEL_SETTINGS lets you keep several workspaces (or point tests at a temp file)
SETTINGS_FILE = Path(os.environ.get("LLM_PANEL_SETTINGS") or (PANEL_ROOT / "panel.yaml"))
EXPERIMENTS_DIR = PANEL_ROOT / "experiments"

DEFAULT_SETTINGS: Dict[str, Any] = {
    # where the two repos live (relative paths are resolved from the panel folder)
    "data_pipeline_dir": "../data-pipeline",
    "evaluation_dir": "../evaluation",
    # python interpreter per tool: each repo can keep its own venv/conda env
    "python": {
        "data": sys.executable,
        "perplexity": sys.executable,
        "eval": sys.executable,
        "train": sys.executable,
    },
    # command used to launch Ludwig (pip install "ludwig[llm]")
    "ludwig_command": "ludwig",
    # every run gets its own folder here
    "runs_dir": "runs",
    # extra folders scanned on the Compare page for existing evaluation results
    "extra_eval_results_dirs": [],
    # environment variables passed to every job
    "env": {
        "HF_HUB_OFFLINE": "",
        "TRANSFORMERS_OFFLINE": "",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    },
    # per-stage environment variables
    "stage_env": {
        "data": {"THAI_NER_PATH": "model/NERmodel", "FT_POS_THRESHOLD": "0.95"},
    },
}


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Settings:
    raw: Dict[str, Any]
    path: Path = SETTINGS_FILE

    def _dir(self, key: str) -> Path:
        p = Path(os.path.expanduser(str(self.raw[key])))
        return p if p.is_absolute() else (PANEL_ROOT / p).resolve()

    @property
    def data_pipeline_dir(self) -> Path:
        return self._dir("data_pipeline_dir")

    @property
    def evaluation_dir(self) -> Path:
        return self._dir("evaluation_dir")

    @property
    def runs_dir(self) -> Path:
        d = self._dir("runs_dir")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def python_for(self, stage: str) -> str:
        return str(self.raw.get("python", {}).get(stage) or sys.executable)

    def env_for(self, stage: str) -> Dict[str, str]:
        env = {k: str(v) for k, v in (self.raw.get("env") or {}).items() if v not in (None, "")}
        env.update({k: str(v) for k, v in ((self.raw.get("stage_env") or {}).get(stage) or {}).items()
                    if v not in (None, "")})
        return env

    @property
    def extra_eval_dirs(self) -> List[Path]:
        out = []
        for p in self.raw.get("extra_eval_results_dirs") or []:
            q = Path(os.path.expanduser(str(p)))
            out.append(q if q.is_absolute() else (PANEL_ROOT / q).resolve())
        return out

    def save(self) -> None:
        self.path.write_text(yaml.safe_dump(self.raw, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_settings(path: Path = SETTINGS_FILE) -> Settings:
    raw: Dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings(_deep_merge(DEFAULT_SETTINGS, raw), path)


# ------------------------------------------------------------------ experiments
_SLUG = re.compile(r"[^0-9A-Za-z฀-๿_.-]+")


def slugify(name: str) -> str:
    s = _SLUG.sub("-", name.strip()).strip("-.")
    return s or "experiment"


def list_experiments(d: Path = EXPERIMENTS_DIR) -> List[str]:
    d.mkdir(parents=True, exist_ok=True)
    return sorted(p.stem for p in d.glob("*.yaml"))


def load_experiment(name: str, d: Path = EXPERIMENTS_DIR) -> Dict[str, Any]:
    p = d / f"{name}.yaml"
    if not p.exists():
        return {"name": name, "notes": "", "data": {}, "perplexity": {}, "train": {}, "eval": {}}
    exp = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    for k in ("data", "perplexity", "train", "eval"):
        exp.setdefault(k, {})
    exp.setdefault("name", name)
    exp.setdefault("notes", "")
    return exp


def save_experiment(exp: Dict[str, Any], d: Path = EXPERIMENTS_DIR) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    name = slugify(exp.get("name") or "experiment")
    exp["name"] = name
    p = d / f"{name}.yaml"
    p.write_text(yaml.safe_dump(exp, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def resolve_in(base: Path, value: Optional[str]) -> Optional[str]:
    """Resolve a user path: ~ expands, relative paths are taken from `base`."""
    if not value:
        return value
    p = Path(os.path.expanduser(str(value)))
    return str(p if p.is_absolute() else (base / p).resolve())
