"""Background job runner.

Started as `python -m core.runner <job_dir>` in its own process session, so jobs
keep running when the Streamlit page reloads or the browser tab closes. It runs
the steps in job.json one after another, streams their output to log.txt and
keeps the status in job.json up to date.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_child: subprocess.Popen | None = None
_cancelled = False


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write(job_dir: Path, job: dict) -> None:
    tmp = job_dir / "job.json.tmp"
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, job_dir / "job.json")


def _on_signal(signum, frame):  # noqa: ARG001
    global _cancelled
    _cancelled = True
    if _child is not None and _child.poll() is None:
        try:
            _child.terminate()
        except Exception:
            pass


def run(job_dir: Path) -> int:
    global _child
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    # real argv (may contain API keys) lives only in .argv.json; read it and delete it
    private_path = job_dir / ".argv.json"
    private = None
    if private_path.exists():
        try:
            private = json.loads(private_path.read_text(encoding="utf-8"))
        finally:
            private_path.unlink(missing_ok=True)
    exec_argv = [s["argv"] for s in (private or job["steps"])]
    job["status"] = "running"
    job["started"] = _now()
    job["runner_pid"] = os.getpid()
    _write(job_dir, job)

    signal.signal(signal.SIGTERM, _on_signal)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _on_signal)

    rc = 0
    with open(job_dir / "log.txt", "a", encoding="utf-8", buffering=1) as log:
        for i, step in enumerate(job["steps"]):
            if _cancelled or (job_dir / "CANCEL").exists():
                break
            log.write(f"\n===== [{_now()}] step {i + 1}/{len(job['steps'])}: {step.get('title', '')}\n")
            log.write("$ " + " ".join(step.get("display_argv") or step["argv"]) + "\n")
            log.write(f"(cwd: {step['cwd']})\n\n")
            env = os.environ.copy()
            env.update(step.get("env") or {})
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("PYTHONIOENCODING", "utf-8")
            job["current_step"] = i
            _write(job_dir, job)
            try:
                _child = subprocess.Popen(
                    exec_argv[i], cwd=step["cwd"], env=env,
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                )
                rc = _child.wait()
            except FileNotFoundError as e:
                log.write(f"\n[panel] command not found: {e}\n")
                rc = 127
            except Exception as e:  # pragma: no cover - defensive
                log.write(f"\n[panel] failed to start step: {e}\n")
                rc = 1
            step["returncode"] = rc
            step["ended"] = _now()
            _write(job_dir, job)
            log.write(f"\n===== step {i + 1} exited with code {rc}\n")
            if rc != 0:
                break

    if _cancelled or (job_dir / "CANCEL").exists():
        job["status"] = "cancelled"
    else:
        job["status"] = "succeeded" if rc == 0 else "failed"
    job["returncode"] = rc
    job["ended"] = _now()
    _write(job_dir, job)
    return rc


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1])))
