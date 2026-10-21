"""Shared fixtures: a tiny, randomly initialised Llama model saved to disk.

Built locally (no Hub download) so the tests run offline and in a few seconds.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>"
    "{% if enable_thinking is defined and not enable_thinking %}<think></think>{% endif %}"
    "{% endif %}"
)


def build_tiny_llama(out_dir: Path) -> Path:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    corpus = [
        "ต่อไปนี้เป็นข้อสอบปรนัยจงเลือกคำตอบที่ถูกต้องที่สุด",
        "คำถาม: ประเทศไทยมีกี่จังหวัด A. 76 B. 77 C. 78 D. 79 คำตอบ: B",
        "Question: What is two plus two? A. 3 B. 4 C. 5 D. 6 Answer: B",
        "The quick brown fox jumps over the lazy dog. " * 3,
        "ภาษาไทยเป็นภาษาที่มีวรรณยุกต์ การตัดคำภาษาไทยจึงยาก",
    ] * 20
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus, trainer)
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>"
    )
    hf_tok.chat_template = CHAT_TEMPLATE
    hf_tok.model_max_length = 512

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=len(hf_tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        bos_token_id=hf_tok.bos_token_id,
        eos_token_id=hf_tok.eos_token_id,
        pad_token_id=hf_tok.pad_token_id,
    )
    model = LlamaForCausalLM(cfg).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    hf_tok.save_pretrained(out_dir)
    return out_dir


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    return build_tiny_llama(tmp_path_factory.mktemp("tiny_llama"))
