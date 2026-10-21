"""Regression tests for the 2026-10 evaluation fixes."""
import torch

from src.model import HFModel, OpenAIModel, VLLMModel, _normalize_thinking
from src.prompt_process.postprocess import PostProcessor


# ---------- answer extraction ----------
def test_boxed_ignores_think_and_takes_last():
    text = "<think>maybe \\boxed{A} ... no</think>The answer is \\boxed{C}"
    assert PostProcessor(text).extract_boxed_number() == "C"


def test_boxed_normalizes_label():
    assert PostProcessor("\\boxed{B. สี่}").extract_boxed_number() == "B"
    assert PostProcessor("\\boxed{ \\text{a} }").extract_boxed_number() == "a"
    assert PostProcessor("no answer").extract_boxed_number() is None
    assert PostProcessor(None).output() == ""


# ---------- thinking flag ----------
def test_normalize_thinking():
    assert _normalize_thinking(True) == "True"
    assert _normalize_thinking("False") == "False"
    assert _normalize_thinking("no_chat_template") == "no_chat_template"
    assert _normalize_thinking(None) is None


def test_hf_chat_template_has_generation_prompt(tiny_model_dir):
    m = HFModel(str(tiny_model_dir))
    on = m.format_instructions("Q", thinking_mode="True")
    off = m.format_instructions("Q", thinking_mode="False")
    plain = m.format_instructions("Q", thinking_mode=None)
    raw = m.format_instructions("Q", thinking_mode="no_chat_template")
    for t in (on, off, plain):
        assert t.rstrip().endswith("<|assistant|>") or t.rstrip().endswith("<think></think>")
    assert off.endswith("<think></think>") and not on.endswith("<think></think>")
    assert raw.endswith("The answer is: ")


# ---------- pt-mode batching ----------
def test_get_logits_batch_invariant(tiny_model_dir):
    m = HFModel(str(tiny_model_dir))
    short = "คำถาม: ประเทศไทย คำตอบ:"
    long = "ต่อไปนี้เป็นข้อสอบปรนัย " * 5 + "Question: two plus two Answer:"
    alone, _ = m.get_logits([short])
    batched, _ = m.get_logits([short, long])
    # bf16 weights -> tiny numeric noise only
    assert torch.allclose(alone[0], batched[0], atol=2e-3), "padding changed the short prompt's logits"

    # the old behaviour (right padding, logits[:, -1]) reads a pad position for the short prompt
    m.tokenizer.padding_side = "right"
    enc = m.tokenizer([short, long], padding="longest", return_tensors="pt")
    old = m.model(enc["input_ids"], attention_mask=enc["attention_mask"]).logits[:, -1, :].float()
    m.tokenizer.padding_side = "left"
    assert (old[0] - alone[0]).abs().max() > 1e-2


def test_get_logits_returns_raw_logits(tiny_model_dir):
    m = HFModel(str(tiny_model_dir))
    logits, _ = m.get_logits(["hello"])
    # raw logits do not sum to 1 (previously a softmax over the vocab was returned)
    assert not torch.allclose(logits.softmax(-1).sum(), logits.sum())


# ---------- API wrappers ----------
def test_api_format_instructions_without_choices():
    assert OpenAIModel("sk-test", "gpt-x").format_instructions("Q", thinking_mode="True")[0]["role"] == "user"
    assert VLLMModel("http://127.0.0.1:9", "m").format_instructions("Q", thinking_mode=None)[0]["role"] == "user"


def test_vllm_connection_failure_detected():
    assert VLLMModel("http://127.0.0.1:9", "m", api_key="k").test_connection() is False


# ---------- end-to-end smoke ----------
def _tiny_mc_df():
    import pandas as pd
    from src.prompt_process.prompt_generator import restructure_dataframe

    df = pd.DataFrame({
        "question": ["ข้อใดคือผลบวกของ 2 และ 2", "ประเทศไทยมีกี่จังหวัด", "สีของท้องฟ้า"],
        "choices": [["3", "4", "5", "6"], ["76", "77", "78", "79"], ["ฟ้า", "แดง", "เขียว", "ดำ"]],
        "answer": [1, 1, 0],
    })
    return restructure_dataframe(df)


def test_eval_pt_batch_matches_batch_one(tiny_model_dir, tmp_path):
    from src.evaluator.evaluator import Evaluator

    ev = Evaluator(HFModel(str(tiny_model_dir)))
    df = _tiny_mc_df()
    common = dict(dev_df=df, task_name="mmlu_thai", exam_type="t", num_shots=1, model_name="tiny")
    r1 = ev.eval_pt(test_df=df, batch_size=1, save_result_dir=str(tmp_path / "b1"), current_time="x", **common)
    r3 = ev.eval_pt(test_df=df, batch_size=3, save_result_dir=str(tmp_path / "b3"), current_time="x", **common)
    assert list(r1["predict"]) == list(r3["predict"])
    probs = r3[[c for c in r3.columns if c.startswith("prob_")]].to_numpy()
    assert abs(probs.sum(axis=1) - 1).max() < 1e-3  # renormalised over the choices once


def test_eval_it_runs_without_choices_error(tiny_model_dir, tmp_path):
    from src.evaluator.evaluator import Evaluator

    ev = Evaluator(HFModel(str(tiny_model_dir)))
    df = _tiny_mc_df().iloc[:2]
    out = ev.eval_it(test_df=df, dev_df=df, task_name="mmlu_thai", exam_type="t", num_shots=0,
                     batch_size=2, save_result_dir=str(tmp_path), model_name="tiny", current_time="x",
                     thinking="no_chat_template")
    assert len(out) == 2 and "correctness" in out.columns
