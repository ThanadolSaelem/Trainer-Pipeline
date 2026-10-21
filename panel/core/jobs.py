"""Create, launch, list and cancel background jobs.

A job is a folder:  runs/<stage>/<timestamp>_<name>/
    job.json   what to run (steps), the experiment snapshot, and live status
    log.txt    combined stdout/stderr of every step
    ...        whatever the tools write (outputs are pointed into this folder)
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PANEL_ROOT, slugify

STAGES = ("data", "perplexity", "train", "eval")
ACTIVE = ("queued", "running")


@dataclass
class Step:
    title: str
    argv: List[str]
    cwd: str
    env: Dict[str, str]
    display_argv: Optional[List[str]] = None  # redacted form for logs/UI

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "argv": [str(a) for a in self.argv],
            "display_argv": [str(a) for a in (self.display_argv or self.argv)],
            "cwd": str(self.cwd),
            "env": {k: str(v) for k, v in self.env.items()},
        }


def new_job_dir(runs_dir: Path, stage: str, name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = runs_dir / stage / f"{ts}_{slugify(name)}"
    n = 1
    while d.exists():
        n += 1
        d = runs_dir / stage / f"{ts}_{slugify(name)}-{n}"
    d.mkdir(parents=True)
    return d


def write_job(job_dir: Path, stage: str, name: str, steps: List[Step],
              snapshot: Dict[str, Any], outputs: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    job = {
        "id": job_dir.name,
        "stage": stage,
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "status": "queued",
        "steps": [s.to_dict() for s in steps],
        # secrets are never stored: the job keeps the redacted argv only
        "snapshot": snapshot,
        "outputs": outputs or {},
    }
    # job.json must not contain secrets: store the redacted argv in place of the real one,
    # the real argv is handed to the runner through a private file that is deleted on start
    private = [s.to_dict() for s in steps]
    for s in job["steps"]:
        s["argv"] = s["display_argv"]
    (job_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    (job_dir / ".argv.json").write_text(json.dumps(private, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(job_dir / ".argv.json", 0o600)
    except OSError:
        pass
    return job


def launch(job_dir: Path) -> int:
    """Start the runner detached from Streamlit; returns the runner pid."""
    log = open(job_dir / "runner.log", "a", encoding="utf-8")
    kwargs: Dict[str, Any] = {}
    if os.name == "nt":  # pragma: no cover - Windows
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True  # own process group: cancel kills the whole tree
    p = subprocess.Popen(
        [sys.executable, "-m", "core.runner", str(job_dir)],
        cwd=str(PANEL_ROOT), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kwargs,
    )
    return p.pid


def submit(runs_dir: Path, stage: str, name: str, build_fn, snapshot: Dict[str, Any]) -> Path:
    """Create a job folder, let `build_fn(job_dir) -> (steps, outputs)` describe the work, start it."""
    job_dir = new_job_dir(runs_dir, stage, name)
    try:
        steps, outputs = build_fn(job_dir)
    except Exception:
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    write_job(job_dir, stage, name, steps, snapshot, outputs)
    launch(job_dir)
    return job_dir


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        import psutil  # type: ignore
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    except Exception:
        return False


def read_job(job_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    job["dir"] = str(job_dir)
    # a runner that died without updating its status (machine slept, killed, ...)
    if job.get("status") == "running" and not _pid_alive(job.get("runner_pid")):
        job["status"] = "lost"
    return job


def list_jobs(runs_dir: Path, stage: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    stages = [stage] if stage else list(STAGES)
    dirs: List[Path] = []
    for st in stages:
        d = runs_dir / st
        if d.is_dir():
            dirs += [p for p in d.iterdir() if (p / "job.json").exists()]
    dirs.sort(key=lambda p: p.name, reverse=True)
    out = []
    for d in dirs[:limit]:
        j = read_job(d)
        if j:
            out.append(j)
    return out


def cancel(job_dir: Path) -> None:
    (job_dir / "CANCEL").write_text("1", encoding="utf-8")
    job = read_job(job_dir) or {}
    pid = job.get("runner_pid")
    if not pid:
        return
    try:
        if os.name == "nt":  # pragma: no cover
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], check=False)
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def tail(path: Path, max_bytes: int = 40_000) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    # tqdm writes \r progress updates: keep only the last state of each line
    lines = [ln.split("\r")[-1] for ln in text.split("\n")]
    return "\n".join(lines)
