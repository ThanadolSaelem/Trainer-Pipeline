"""Regression tests for the 2026-10 data-pipeline fixes."""
import json
import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # Pipeline_clean imports `src.f_text_cleaner` relative to the repo root

import Pipeline_clean as pc  # noqa: E402
import Perplexity as ppl  # noqa: E402


def _write_jsonl(path: Path, n: int):
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"id": i, "text": f"ข้อความทดสอบที่ {i} สวัสดีครับ"}, ensure_ascii=False) + "\n")


def _count(path: Path) -> int:
    return sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0


# ---------------- Pipeline_clean ----------------
@pytest.mark.parametrize("single", [True, False])
def test_no_duplicate_output_and_clean_filenames(tmp_path, single):
    data = tmp_path / "data"
    data.mkdir()
    _write_jsonl(data / "a.jsonl", 25)
    out = tmp_path / "out"
    argv = ["-i", str(data), "-o", str(out), "--chunk-size", "10", "--workers", "1",
            "--clean-steps", "html", "--no-fasttext"]
    if single:
        argv.append("--single-process")
    assert pc.main(argv) == 0
    good = out / "GOOD_data.jsonl"
    bad = out / "BAD_data.jsonl"
    assert good.exists() and bad.exists(), sorted(p.name for p in out.iterdir())
    # 25 input rows across 3 chunks -> exactly 25 output rows (single-process used to duplicate)
    assert _count(good) + _count(bad) == 25
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["stats"]["processed"] == 25


def test_single_process_rerun_does_not_merge_stale_parts(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_jsonl(data / "a.jsonl", 12)
    out = tmp_path / "out"
    argv = ["-i", str(data), "-o", str(out), "--chunk-size", "5", "--single-process", "--no-fasttext"]
    pc.main(argv)
    pc.main(argv)
    assert _count(out / "GOOD_data.jsonl") == 12


def test_debug_first_n_limits_rows_and_isolates_checkpoint(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_jsonl(data / "a.jsonl", 30)
    out = tmp_path / "out"
    base = ["-i", str(data), "-o", str(out), "--chunk-size", "10", "--workers", "1", "--no-fasttext"]
    pc.main(base + ["--debug-first-n", "4"])
    assert _count(out / "GOOD_data.jsonl") == 4
    # a full run afterwards must not skip part 0 because of the debug run's done-marker
    pc.main(base)
    assert _count(out / "GOOD_data.jsonl") == 30


def test_addr_options_reach_options(tmp_path, monkeypatch):
    captured = {}
    real = pc.Options

    def spy(**kw):
        captured.update(kw)
        return real(**kw)

    monkeypatch.setattr(pc, "Options", spy)
    data = tmp_path / "a.jsonl"
    _write_jsonl(data, 1)
    pc.main(["-i", str(data), "-o", str(tmp_path / "o"), "--single-process", "--no-fasttext",
             "--addr-gazetteer", "gz.json", "--addr-tag", "[ที่อยู่]", "--no-addr-keep-province"])
    assert captured["addr_gazetteer"] == "gz.json"
    assert captured["addr_tag"] == "[ที่อยู่]"
    assert captured["addr_keep_province"] is False


def test_public_figure_skip_enabled_when_lists_exist(tmp_path):
    data = tmp_path / "a.jsonl"
    _write_jsonl(data, 1)
    pc.main(["-i", str(data), "-o", str(tmp_path / "o"), "--single-process", "--no-fasttext"])
    assert pc.public_figure_skip is True  # corpus/ ships the name lists
    pc.main(["-i", str(data), "-o", str(tmp_path / "o2"), "--single-process", "--no-fasttext",
             "--no-public-figure-skip"])
    assert pc.public_figure_skip is False


# ---------------- Perplexity ----------------
LONG_TEXT = "The quick brown fox jumps over the lazy dog. ภาษาไทยเป็นภาษาที่มีวรรณยุกต์ " * 12


@pytest.mark.parametrize("overlap_ratio", [0.0, 0.25, 0.5])
def test_overlap_tokens_scored_once(tiny_model_dir, overlap_ratio):
    _, tok = ppl.load_model_and_tokenizer(str(tiny_model_dir))
    n_ids = len(tok.encode(ppl.sanitize_text(LONG_TEXT), add_special_tokens=True))
    res = ppl.compute_ppl(str(tiny_model_dir), LONG_TEXT, context_length=64,
                          overlap_ratio=overlap_ratio, is_file=False)
    assert n_ids > 64 * 2, "text must span several windows for this test to mean anything"
    # every token except the very first is predicted exactly once, whatever the overlap
    assert res["tokens_evaluated"] == n_ids - 1


def test_micro_ppl_is_token_level():
    # two texts: PPL 2 over 1 token and PPL 8 over 3 tokens
    got = ppl.micro_ppl([2.0, 8.0], [1, 3])
    want = math.exp((math.log(2) * 1 + math.log(8) * 3) / 4)
    assert got == pytest.approx(want)
    assert got != pytest.approx((2 * 1 + 8 * 3) / 4)  # the old weighted-average formula


def test_short_text_not_padded_to_context(tiny_model_dir):
    model, tok = ppl.load_model_and_tokenizer(str(tiny_model_dir))
    enc = ppl.encode_with_overlap(tok, ["สวัสดี"], max_length=512, overlap_tokens=0, use_chat_template=False)
    assert enc[0][0]["input_ids"].shape[1] < 20


def test_model_loaded_once(tiny_model_dir):
    a = ppl.load_model_and_tokenizer(str(tiny_model_dir))
    b = ppl.load_model_and_tokenizer(str(tiny_model_dir))
    assert a[0] is b[0]


def test_per_text_details_and_folder_jsonl(tiny_model_dir, tmp_path):
    folder = tmp_path / "texts"
    folder.mkdir()
    (folder / "a.jsonl").write_text(
        "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in ["hello world", "ภาษาไทย"]),
        encoding="utf-8")
    out = tmp_path / "per_text.jsonl"
    summary = ppl.process_folder(str(folder), str(tiny_model_dir), context_length=64,
                                 verbose=False, per_text_jsonl=str(out), top_k_tokens=3)
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and rows[0]["file"] == "a.jsonl"
    assert rows[0]["chars_per_token"] > 0 and len(rows[0]["worst_tokens"]) <= 3
    assert summary["PPL_micro_folder"] > 0


def test_instruct_heuristic():
    assert ppl.looks_like_instruct_model("google/gemma-3-1b-it")
    assert ppl.looks_like_instruct_model("Qwen/Qwen2.5-7B-Instruct")
    assert not ppl.looks_like_instruct_model("gpt2-with-italics")  # contains "it" but is not an instruct model
