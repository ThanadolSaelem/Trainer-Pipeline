"""Translate experiment values into job steps for each stage."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from . import schema as S
from .config import PANEL_ROOT, Settings, resolve_in
from .jobs import Step
from .ludwig_cfg import effective_config, to_yaml

BuildFn = Callable[[Path], Tuple[List[Step], Dict[str, str]]]


def _ludwig_argv(settings: Settings) -> List[str]:
    cmd = str(settings.raw.get("ludwig_command") or "").strip()
    if cmd and cmd != "ludwig":
        return cmd.split()
    # run the CLI module with the train interpreter, so no PATH setup is needed
    return [settings.python_for("train"), "-m", "ludwig.cli"]


# ------------------------------------------------------------------ validation
def validate(stage: str, v: Dict[str, Any], settings: Settings) -> List[str]:
    errs: List[str] = []
    if stage == "data":
        if not v.get("input"):
            errs.append("ยังไม่ได้เลือกไฟล์/โฟลเดอร์ input")
        elif not Path(resolve_in(settings.data_pipeline_dir, v["input"])).exists():
            errs.append(f"ไม่พบ input: {v['input']}")
        if not (settings.data_pipeline_dir / "Pipeline_clean.py").exists():
            errs.append(f"ไม่พบ Pipeline_clean.py ใน {settings.data_pipeline_dir} (ตั้งค่าที่หน้า Settings)")
    elif stage == "perplexity":
        if not v.get("model"):
            errs.append("ยังไม่ได้ระบุโมเดล")
        if not v.get("target"):
            errs.append("ยังไม่ได้เลือกไฟล์/โฟลเดอร์ที่จะวัด")
        else:
            target = Path(resolve_in(settings.data_pipeline_dir, v["target"]))
            if not target.exists():
                errs.append(f"ไม่พบ: {v['target']}")
            elif target.is_dir() and (any(target.glob("tmp_*/part_*.jsonl")) or (target / "summary.txt").exists()):
                # folder mode reads every .jsonl/.txt below it: the pipeline's tmp parts and summary
                # would be measured again, so each text would count several times
                goods = sorted(target.glob("GOOD_*.jsonl"))
                hint = f" เช่น {goods[0]}" if goods else ""
                errs.append("โฟลเดอร์นี้เป็นผลลัพธ์ของ Data Cleaning ซึ่งมีไฟล์ชั่วคราว tmp_*/ และ summary "
                            f"จะถูกวัดซ้ำ ให้เลือกไฟล์ GOOD_*.jsonl โดยตรงแทน{hint}")
    elif stage == "eval":
        if not v.get("model_path"):
            errs.append("ยังไม่ได้ระบุโมเดล")
        if v.get("source") == "vllm" and not v.get("vllm_url"):
            errs.append("โหมด vLLM ต้องระบุ URL")
        if v.get("source") == "openai" and not (v.get("openai_key") or os.environ.get("OPENAI_API_KEY")):
            errs.append("โหมด OpenAI ต้องมี API key (ในฟอร์มหรือ env OPENAI_API_KEY)")
        if v.get("source") in ("vllm", "openai") and v.get("mode") == "pt":
            errs.append("โมเดลผ่าน API รองรับเฉพาะโหมด it")
        if not v.get("data"):
            errs.append("ยังไม่ได้เลือกชุดข้อสอบ")
        if not (settings.evaluation_dir / "src" / "run.py").exists():
            errs.append(f"ไม่พบ src/run.py ใน {settings.evaluation_dir} (ตั้งค่าที่หน้า Settings)")
    elif stage == "train":
        if not v.get("use_raw_yaml") and not v.get("base_model"):
            errs.append("ยังไม่ได้ระบุ base model")
        if not v.get("dataset"):
            errs.append("ยังไม่ได้เลือกไฟล์ข้อมูลเทรน")
        elif not Path(os.path.expanduser(v["dataset"])).exists():
            errs.append(f"ไม่พบไฟล์ข้อมูล: {v['dataset']}")
        try:
            effective_config(v)
        except Exception as e:
            errs.append(f"Ludwig YAML ไม่ถูกต้อง: {e}")
    return errs


# ------------------------------------------------------------------ builders
def data_job(v: Dict[str, Any], settings: Settings) -> BuildFn:
    vals = S.merged(S.PIPELINE, v)
    cwd = settings.data_pipeline_dir

    def build(job_dir: Path):
        out = job_dir / "output"
        argv = [settings.python_for("data"), "Pipeline_clean.py", *S.build_argv(S.PIPELINE, vals), "--out", str(out)]
        step = Step("Clean data", argv, str(cwd), settings.env_for("data"), S.redact(argv, S.PIPELINE))
        return [step], {"output_dir": str(out)}

    return build


def perplexity_job(v: Dict[str, Any], settings: Settings) -> BuildFn:
    vals = S.merged(S.PERPLEXITY, v)
    cwd = settings.data_pipeline_dir
    target = resolve_in(cwd, vals["target"])

    def build(job_dir: Path):
        summary = job_dir / "summary.json"
        per_text = job_dir / "per_text.jsonl"
        py = settings.python_for("perplexity")
        opts = S.build_argv(S.PERPLEXITY, vals)
        if Path(target).is_dir():
            argv = [py, "Perplexity.py", "folder", "--folder", target, *opts,
                    "--output_json", str(summary), "--per_text_jsonl", str(per_text)]
        else:
            argv = [py, "Perplexity.py", "--file", target, *opts,
                    "--output_json", str(summary), "--per_text_jsonl", str(per_text)]
        step = Step("Perplexity", argv, str(cwd), settings.env_for("perplexity"))
        return [step], {"summary": str(summary), "per_text": str(per_text), "target": target}

    return build


def eval_job(v: Dict[str, Any], settings: Settings) -> BuildFn:
    vals = S.merged(S.EVALUATION, v)
    cwd = settings.evaluation_dir
    source = vals.get("source", "hf")
    skip = {"hf": ["vllm_url", "api_key", "openai_key"],
            "vllm": ["openai_key"],
            "openai": ["vllm_url", "api_key"]}[source]
    if vals.get("thinking_mode") in (None, "None", ""):
        skip.append("thinking_mode")  # src.run accepts only the Python None default, not the string
    if source == "openai" and not vals.get("openai_key"):
        vals["openai_key"] = os.environ.get("OPENAI_API_KEY", "")
    model_path = vals["model_path"]
    if source == "hf":
        local = resolve_in(PANEL_ROOT, model_path)
        if local and Path(local).exists():
            vals["model_path"] = local

    def build(job_dir: Path):
        results = job_dir / "results"
        argv = [settings.python_for("eval"), "-m", "src.run", *S.build_argv(S.EVALUATION, vals, skip=skip),
                "--save-result-dir", str(results), "--project-name", "panel"]
        step = Step("Evaluate", argv, str(cwd), settings.env_for("eval"), S.redact(argv, S.EVALUATION))
        return [step], {"results_dir": str(results)}

    return build


def train_job(v: Dict[str, Any], settings: Settings, experiment_name: str) -> BuildFn:
    cfg = effective_config(v)
    dataset = os.path.expanduser(v["dataset"])

    def build(job_dir: Path):
        cfg_path = job_dir / "ludwig_config.yaml"
        cfg_path.write_text(to_yaml(cfg), encoding="utf-8")
        ludwig_dir = job_dir / "ludwig"
        hf_dir = job_dir / "hf_model"
        env = settings.env_for("train")
        steps = [Step("Ludwig train",
                      [*_ludwig_argv(settings), "train", "--config", str(cfg_path), "--dataset", dataset,
                       "--output_directory", str(ludwig_dir), "--experiment_name", experiment_name,
                       "--skip_save_processed_input"],
                      str(job_dir), env)]
        if cfg.get("base_model"):
            steps.append(Step("Export Hugging Face model",
                              [settings.python_for("train"), "-m", "core.export_hf", "--ludwig-dir", str(ludwig_dir),
                               "--base-model", str(cfg["base_model"]), "--out", str(hf_dir)],
                              str(PANEL_ROOT), env))
        return steps, {"config": str(cfg_path), "ludwig_dir": str(ludwig_dir), "hf_model": str(hf_dir)}

    return build
