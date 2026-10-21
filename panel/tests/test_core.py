import json
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import commands as C  # noqa: E402
from core import jobs as J  # noqa: E402
from core import results as R  # noqa: E402
from core import schema as S  # noqa: E402
from core.config import Settings, DEFAULT_SETTINGS, _deep_merge  # noqa: E402
from core.ludwig_cfg import build_ludwig_config, effective_config  # noqa: E402


def make_settings(tmp_path, **over) -> Settings:
    raw = _deep_merge(DEFAULT_SETTINGS, {"runs_dir": str(tmp_path / "runs"), **over})
    return Settings(raw, tmp_path / "panel.yaml")


def wait(job_dir: Path, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = J.read_job(job_dir)
        if j and j["status"] not in J.ACTIVE:
            return j
        time.sleep(0.2)
    raise TimeoutError(J.tail(job_dir / "log.txt"))


# ------------------------------------------------------------------ schema
def test_build_argv_pipeline_defaults_are_minimal():
    argv = S.build_argv(S.PIPELINE, {"input": "data/"})
    assert argv[:2] == ["--input", "data/"]
    assert "--clean-steps" in argv and argv[argv.index("--clean-steps") + 1] == "html,emoji,emoticon"
    assert "--debug-first-n" not in argv  # 0 = off -> omitted
    assert "--thai-ner-full" not in argv


def test_build_argv_bool_false_flag_and_multi_space_joined():
    argv = S.build_argv(S.PIPELINE, {"input": "x", "resume": False, "thai_ner_full": True})
    assert "--no-resume" in argv and "--thai-ner-full" in argv
    ev = S.build_argv(S.EVALUATION, {"model_path": "m", "data": ["xnli", "xcopa"]})
    i = ev.index("--data")
    assert ev[i + 1:i + 3] == ["xnli", "xcopa"]


def test_redact_hides_secrets():
    argv = S.build_argv(S.EVALUATION, {"model_path": "gpt", "openai_key": "sk-secret"})
    red = S.redact(argv, S.EVALUATION)
    assert "sk-secret" in argv and "sk-secret" not in red and "***" in red


# ------------------------------------------------------------------ commands
def test_eval_job_skips_none_thinking_and_irrelevant_source_flags(tmp_path):
    st = make_settings(tmp_path)
    build = C.eval_job({"model_path": "m", "source": "hf", "vllm_url": "http://x", "thinking_mode": "None"}, st)
    steps, outs = build(tmp_path)
    argv = steps[0].argv
    assert "--vllm" not in argv and "--thinking-mode" not in argv
    assert argv[argv.index("--save-result-dir") + 1] == str(tmp_path / "results")


def test_eval_job_secret_not_in_display(tmp_path):
    st = make_settings(tmp_path)
    steps, _ = C.eval_job({"model_path": "gpt-x", "source": "openai", "openai_key": "sk-123", "mode": "it"}, st)(tmp_path)
    assert "sk-123" in steps[0].argv and "sk-123" not in steps[0].display_argv


def test_perplexity_job_folder_vs_file(tmp_path):
    st = make_settings(tmp_path)
    d = tmp_path / "texts"
    d.mkdir()
    (d / "a.jsonl").write_text('{"text":"x"}\n')
    steps, _ = C.perplexity_job({"model": "m", "target": str(d)}, st)(tmp_path)
    assert steps[0].argv[2:4] == ["folder", "--folder"]
    steps, _ = C.perplexity_job({"model": "m", "target": str(d / "a.jsonl")}, st)(tmp_path)
    assert steps[0].argv[2] == "--file"


def test_validate_messages(tmp_path):
    st = make_settings(tmp_path, data_pipeline_dir=str(tmp_path), evaluation_dir=str(tmp_path))
    assert C.validate("data", {"input": ""}, st)
    assert any("API" in e for e in C.validate("eval", {"model_path": "m", "source": "openai", "mode": "pt",
                                                        "data": ["xnli"]}, st))


# ------------------------------------------------------------------ ludwig config
def test_ludwig_config_lora_merge_and_template():
    cfg = build_ludwig_config({"base_model": "b", "input_column": "q", "output_column": "a",
                               "prompt_template": "Q: {q}\\nA:", "adapter": "lora", "lora_r": 16,
                               "merge_adapter": True, "quantization_bits": "none", "epochs": 1})
    assert cfg["input_features"][0]["name"] == "q"
    assert cfg["adapter"]["r"] == 16
    assert cfg["adapter"]["postprocessor"]["merge_adapter_into_base_model"] is True
    assert cfg["prompt"]["template"] == "Q: {q}\nA:"
    assert "quantization" not in cfg


def test_ludwig_raw_yaml_overrides_form():
    cfg = effective_config({"use_raw_yaml": True, "raw_yaml": "model_type: llm\nbase_model: z\n", "base_model": "b"})
    assert cfg["base_model"] == "z"


def test_train_job_writes_config_and_export_step(tmp_path):
    st = make_settings(tmp_path)
    ds = tmp_path / "d.jsonl"
    ds.write_text('{"instruction":"a","output":"b"}\n')
    steps, outs = C.train_job({"base_model": "b", "dataset": str(ds)}, st, "exp")(tmp_path)
    assert yaml.safe_load(Path(outs["config"]).read_text())["base_model"] == "b"
    assert steps[0].argv[1:3] == ["-m", "ludwig.cli"] and "train" in steps[0].argv
    assert steps[1].argv[1:3] == ["-m", "core.export_hf"]


# ------------------------------------------------------------------ jobs (real subprocesses)
def test_job_runs_steps_and_records_status(tmp_path):
    st = make_settings(tmp_path)
    py = sys.executable

    def build(job_dir):
        return [J.Step("one", [py, "-c", "print('hello from step 1')"], str(tmp_path), {}),
                J.Step("two", [py, "-c", "import os; print(os.environ['PANEL_T'])"], str(tmp_path), {"PANEL_T": "env-ok"})], {}

    job_dir = J.submit(st.runs_dir, "data", "smoke", build, {"x": 1})
    job = wait(job_dir)
    log = J.tail(job_dir / "log.txt")
    assert job["status"] == "succeeded", log
    assert "hello from step 1" in log and "env-ok" in log
    assert [s["returncode"] for s in job["steps"]] == [0, 0]
    assert not (job_dir / ".argv.json").exists()


def test_job_failure_stops_following_steps(tmp_path):
    st = make_settings(tmp_path)
    py = sys.executable

    def build(job_dir):
        return [J.Step("bad", [py, "-c", "raise SystemExit(3)"], str(tmp_path), {}),
                J.Step("never", [py, "-c", "print('nope')"], str(tmp_path), {})], {}

    job = wait(J.submit(st.runs_dir, "eval", "fail", build, {}))
    assert job["status"] == "failed" and job["returncode"] == 3
    assert "returncode" not in job["steps"][1]


def test_job_cancel_kills_process_tree(tmp_path):
    st = make_settings(tmp_path)
    py = sys.executable
    marker = tmp_path / "child_done"
    script = (f"import subprocess,sys,time; "
              f"subprocess.Popen([sys.executable,'-c','import time; time.sleep(5); open(r\"{marker}\",\"w\").write(\"x\")']); "
              f"time.sleep(30)")

    def build(job_dir):
        return [J.Step("sleepy", [py, "-c", script], str(tmp_path), {})], {}

    job_dir = J.submit(st.runs_dir, "train", "cancel", build, {})
    t0 = time.time()
    while (J.read_job(job_dir) or {}).get("status") != "running" and time.time() - t0 < 10:
        time.sleep(0.1)
    time.sleep(0.5)
    J.cancel(job_dir)
    job = wait(job_dir, timeout=15)
    assert job["status"] in ("cancelled", "lost")
    time.sleep(6)
    assert not marker.exists(), "grandchild process survived cancel"


def test_secret_never_written_to_job_json(tmp_path):
    st = make_settings(tmp_path)
    py = sys.executable

    def build(job_dir):
        argv = [py, "-c", "import sys; print(len(sys.argv))", "--openai", "sk-zzz"]
        return [J.Step("s", argv, str(tmp_path), {}, S.redact(argv, S.EVALUATION))], {}

    job_dir = J.submit(st.runs_dir, "eval", "secret", build, {})
    wait(job_dir)
    assert "sk-zzz" not in (job_dir / "job.json").read_text()
    assert "sk-zzz" not in (job_dir / "log.txt").read_text()


# ------------------------------------------------------------------ results
def test_eval_result_parsing(tmp_path):
    run = tmp_path / "results" / "tiny_panel"
    (run / "summary").mkdir(parents=True)
    (run / "m3exam").mkdir()
    (run / "summary" / "m3exam_report.json").write_text(json.dumps(
        {"subject": {"math": 0.5, "thai": 1.0}, "level": {"low": "N/A"}, "overall": 0.75}))
    (run / "summary" / "xnli_report.json").write_text(json.dumps({"overall": 0.4}))
    (run / "m3exam" / "default_result.csv").write_text(
        "question,answer_text,predict,correctness,subject\nq1,1,1,1,math\nq2,2,1,0,math\nq3,1,1,1,thai\nq4,3,,0,thai\n")
    runs = R.find_eval_runs([tmp_path])
    assert len(runs) == 1
    tbl = R.overall_table(runs)
    assert tbl.loc[runs[0]["label"], "m3exam"] == 0.75
    sl = R.slice_table(runs, "m3exam")
    assert set(sl["slice"]) == {"math", "thai"}  # "N/A" values skipped
    preds = R.load_predictions(run, "m3exam")
    cm = R.confusion(preds)
    assert cm.loc["2", "1"] == 1 and "∅ (ไม่มีคำตอบ)" in cm.columns
    acc = R.slice_accuracy(preds)
    assert acc["subject"].loc["math", "accuracy"] == 0.5


def test_pipeline_summary_and_sampling(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"stats": {"processed": 3, "Good": 2, "Bad": 1}}))
    good = tmp_path / "GOOD_x.jsonl"
    good.write_text("\n".join(json.dumps({"text": str(i), "__flags": ["mixed_adjacent"]}) for i in range(50)))
    assert R.pipeline_summary(tmp_path)["stats"]["Good"] == 2
    assert R.pipeline_outputs(tmp_path)["good"] == good
    assert len(R.sample_jsonl(good, 10)) == 10
    assert [r["text"] for r in R.sample_jsonl(good, 3, head=True)] == ["0", "1", "2"]
    assert R.flag_counts(good)["mixed_adjacent"] == 50


def test_ludwig_stats_parsing(tmp_path):
    d = tmp_path / "exp_run"
    d.mkdir()
    (d / "training_statistics.json").write_text(json.dumps(
        {"training": {"output": {"loss": [2.0, 1.5]}, "combined": {"loss": [2.0, 1.5]}},
         "validation": {"output": {"loss": [2.1, 1.7]}}}))
    df = R.ludwig_stats(tmp_path)
    assert len(df[(df.metric == "loss") & (df.split == "validation")]) == 2


def test_log_hints_detect_silent_fallbacks():
    from core.ui import log_hints
    log = ("[2026] WARNING [import] f_text_cleaner not available: boom\n"
           "ModuleNotFoundError: No module named 'gdown'\n"
           "[clean-step] unknown step ignored: 'emojii' (known: ...)\n")
    hints = log_hints(log)
    assert any("f_text_cleaner" in h for h in hints)
    assert any("gdown" in h for h in hints)
    assert any("emojii" in h for h in hints)
    assert log_hints("all good") == []
    # optional-dependency notices are not failures
    assert log_hints("Failed to load CPU kernel: No module named 'kernels'. Please pip install kernels") == []


def test_perplexity_refuses_pipeline_output_folder(tmp_path):
    out = tmp_path / "output"
    (out / "tmp_a").mkdir(parents=True)
    (out / "tmp_a" / "part_00000000.Good.jsonl").write_text("{}\n")
    (out / "GOOD_a.jsonl").write_text("{}\n")
    st = make_settings(tmp_path)
    errs = C.validate("perplexity", {"model": "m", "target": str(out)}, st)
    assert any("GOOD_a.jsonl" in e for e in errs)
    assert not C.validate("perplexity", {"model": "m", "target": str(out / "GOOD_a.jsonl")}, st)
