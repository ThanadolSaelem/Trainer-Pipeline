"""Read what the tools wrote, into pandas-friendly structures for the pages."""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

# columns in the evaluation result CSVs that describe a slice of the test set
SLICE_COLUMNS = ("subject", "level", "year", "category", "exam_type", "task", "src")


# ------------------------------------------------------------------ data pipeline
def pipeline_summary(output_dir: Path) -> Dict[str, Any]:
    p = output_dir / "summary.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    # older runs: parse "key: value" lines from summary.txt
    stats: Dict[str, int] = {}
    t = output_dir / "summary.txt"
    if t.exists():
        for line in t.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Za-z_]+):\s*(\d+)$", line.strip())
            if m:
                stats[m.group(1)] = int(m.group(2))
    return {"stats": stats, "options": {}}


def pipeline_outputs(output_dir: Path) -> Dict[str, Optional[Path]]:
    good = sorted(output_dir.glob("GOOD_*.jsonl"))
    bad = sorted(output_dir.glob("BAD_*.jsonl")) or sorted(output_dir.glob("*BAD_*.jsonl"))
    return {"good": good[0] if good else None, "bad": bad[0] if bad else None}


def count_lines(path: Optional[Path]) -> int:
    if not path or not path.exists():
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def sample_jsonl(path: Optional[Path], n: int = 20, seed: int = 0, head: bool = False) -> List[Dict[str, Any]]:
    """Reservoir-sample n records without loading the whole file."""
    if not path or not path.exists():
        return []
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if head and i >= n:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(out) < n:
                out.append(rec)
            elif not head:
                j = rng.randint(0, i)
                if j < n:
                    out[j] = rec
    return out


def flag_counts(path: Optional[Path], flags_key: str = "__flags", limit: int = 200_000) -> Counter:
    c: Counter = Counter()
    if not path or not path.exists() or not flags_key:
        return c
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            if flags_key not in line:
                continue
            try:
                for fl in json.loads(line).get(flags_key) or []:
                    c[fl] += 1
            except json.JSONDecodeError:
                pass
    return c


# ------------------------------------------------------------------ perplexity
def ppl_summary(job_dir: Path) -> Dict[str, Any]:
    p = job_dir / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def ppl_per_text(job_dir: Path) -> pd.DataFrame:
    p = job_dir / "per_text.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "file" not in df.columns:
        df["file"] = ""
    df["worst"] = df.get("worst_tokens", pd.Series([[]] * len(df))).apply(
        lambda ws: " | ".join(f"{w['token']!r} ({w['loss']:.1f})" for w in (ws or [])[:5]))
    df["worst_context"] = df.get("worst_tokens", pd.Series([[]] * len(df))).apply(
        lambda ws: " | ".join(f"…{w.get('context', '')}…" for w in (ws or [])[:3]))
    return df


def worst_token_table(df: pd.DataFrame, top: int = 30) -> pd.DataFrame:
    """Tokens that most often appear among each text's highest-loss tokens."""
    c: Counter = Counter()
    loss_sum: Counter = Counter()
    for ws in df.get("worst_tokens", []):
        for w in ws or []:
            c[w["token"]] += 1
            loss_sum[w["token"]] += w["loss"]
    example: Dict[str, str] = {}
    for ws in df.get("worst_tokens", []):
        for w in ws or []:
            example.setdefault(w["token"], w.get("context", ""))
    rows = [{"token": t, "count": n, "mean_loss": loss_sum[t] / n, "ตัวอย่างบริบท": f"…{example.get(t, '')}…"}
            for t, n in c.most_common(top)]
    return pd.DataFrame(rows)


def text_lookup(target: Path, file: str, index: int, field: Optional[str] = None) -> Optional[str]:
    """Fetch the original text behind a per-text PPL row (jsonl/txt only)."""
    path = target / file if target.is_dir() else target
    if not path.exists():
        return None
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        k = -1
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                k += 1
                if k == index:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        return line
                    if field and field in obj:
                        return str(obj[field])
                    for key in ("text", "content", "body", "output", "instruction"):
                        if key in obj:
                            return str(obj[key])
                    return line
        return None
    if index == 0:
        return path.read_text(encoding="utf-8", errors="replace")[:20000]
    return None


# ------------------------------------------------------------------ evaluation
def find_eval_runs(roots: Iterable[Path]) -> List[Dict[str, Any]]:
    """Every folder that contains summary/<dataset>_report.json, as one 'run'."""
    runs = []
    seen = set()
    for root in roots:
        if not root or not root.exists():
            continue
        for summ in root.rglob("summary"):
            if not summ.is_dir() or summ in seen:
                continue
            reports = sorted(summ.glob("*_report.json"))
            if not reports:
                continue
            seen.add(summ)
            run_dir = summ.parent
            runs.append({"label": _run_label(run_dir), "dir": run_dir, "reports": reports})
    runs.sort(key=lambda r: r["dir"].stat().st_mtime, reverse=True)
    return runs


def _run_label(run_dir: Path) -> str:
    # panel layout: runs/eval/<job>/results/<model>_panel
    parts = run_dir.parts
    if len(parts) >= 3 and parts[-2] == "results":
        return f"{run_dir.name.removesuffix('_panel')} · {parts[-3]}"
    return run_dir.name


def load_reports(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for p in sorted((run_dir / "summary").glob("*_report.json")):
        try:
            out[p.name.removesuffix("_report.json")] = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return out


def overall_table(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    """rows = run, columns = dataset, value = overall accuracy."""
    rows = []
    for r in runs:
        rep = load_reports(r["dir"])
        row = {"run": r["label"]}
        for ds, j in rep.items():
            if isinstance(j.get("overall"), (int, float)):
                row[ds] = j["overall"]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("run") if rows else pd.DataFrame()
    if not df.empty:
        df["mean"] = df.mean(axis=1, numeric_only=True)
    return df


def slice_table(runs: List[Dict[str, Any]], dataset: str) -> pd.DataFrame:
    """Long table: run, slice_type, slice, accuracy (from the nested report JSON)."""
    rows = []
    for r in runs:
        rep = load_reports(r["dir"]).get(dataset) or {}
        for slice_type, vals in rep.items():
            if isinstance(vals, dict):
                for k, acc in vals.items():
                    if isinstance(acc, (int, float)):
                        rows.append({"run": r["label"], "slice_type": slice_type, "slice": str(k), "accuracy": acc})
    return pd.DataFrame(rows)


def load_predictions(run_dir: Path, dataset: str) -> pd.DataFrame:
    d = run_dir / dataset
    frames = []
    for p in sorted(d.glob("*_result.csv")):
        try:
            # labels such as "1".."5" must stay strings, otherwise "1" becomes 1.0 and never matches
            df = pd.read_csv(p, dtype={"predict": str, "answer_text": str})
        except Exception:
            continue
        df["exam_file"] = p.name.removesuffix("_result.csv")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def confusion(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "predict" not in df.columns or "answer_text" not in df.columns:
        return pd.DataFrame()
    pred = df["predict"].fillna("∅ (ไม่มีคำตอบ)").astype(str)
    gold = df["answer_text"].astype(str)
    return pd.crosstab(gold.rename("เฉลย"), pred.rename("โมเดลตอบ"))


def slice_accuracy(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {}
    if df.empty or "correctness" not in df.columns:
        return out
    for col in SLICE_COLUMNS + ("exam_file",):
        if col in df.columns and 1 < df[col].nunique() <= 200:
            g = df.groupby(col)["correctness"].agg(["mean", "count"]).rename(columns={"mean": "accuracy", "count": "n"})
            out[col] = g.sort_values("accuracy")
    return out


# ------------------------------------------------------------------ ludwig
def ludwig_stats(ludwig_dir: Path) -> pd.DataFrame:
    """Loss/metric curves from training_statistics.json (long format)."""
    files = sorted(ludwig_dir.glob("*/training_statistics.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return pd.DataFrame()
    stats = json.loads(files[-1].read_text(encoding="utf-8"))
    rows = []
    for split, feats in stats.items():
        if not isinstance(feats, dict):
            continue
        for feat, metrics in feats.items():
            if not isinstance(metrics, dict):
                continue
            for metric, values in metrics.items():
                if isinstance(values, list):
                    for i, val in enumerate(values):
                        if isinstance(val, (int, float)):
                            rows.append({"split": split, "feature": feat, "metric": metric, "epoch": i + 1, "value": val})
    return pd.DataFrame(rows)
