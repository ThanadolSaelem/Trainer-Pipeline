"""Turn a Ludwig LLM run into a plain Hugging Face model folder the evaluation repo can load.

Ludwig saves weights at <output_directory>/<experiment>_run*/model/model_weights with
save_pretrained(): full weights when the LoRA adapter was merged, otherwise only the
adapter. This script copies (or merges) them into one folder and adds the base model's
tokenizer, which Ludwig does not save there.

    python -m core.export_hf --ludwig-dir RUN/ludwig --base-model NAME --out RUN/hf_model
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def find_weights(ludwig_dir: Path) -> Path:
    cands = sorted(ludwig_dir.glob("*/model/model_weights"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit(f"[export] no model/model_weights found under {ludwig_dir}")
    return cands[-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ludwig-dir", required=True, type=Path)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    weights = find_weights(a.ludwig_dir)
    print(f"[export] weights: {weights}")
    a.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    if (weights / "adapter_config.json").exists():
        print("[export] adapter only (not merged): merging LoRA into the base model")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(a.base_model)
        merged = PeftModel.from_pretrained(base, str(weights)).merge_and_unload()
        merged.save_pretrained(a.out)
    else:
        for item in weights.iterdir():
            dst = a.out / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)

    AutoTokenizer.from_pretrained(a.base_model).save_pretrained(a.out)
    print(f"[export] Hugging Face model ready: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
