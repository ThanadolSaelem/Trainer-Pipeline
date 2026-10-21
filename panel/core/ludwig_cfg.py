"""Build a Ludwig LLM fine-tuning config from the panel's form values."""
from __future__ import annotations

from typing import Any, Dict

import yaml


def build_ludwig_config(v: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "model_type": "llm",
        "base_model": v["base_model"],
        "input_features": [{"name": v.get("input_column") or "instruction", "type": "text"}],
        "output_features": [{"name": v.get("output_column") or "output", "type": "text"}],
        "trainer": {
            "type": "finetune",
            "epochs": int(v.get("epochs") or 3),
            "learning_rate": float(v.get("learning_rate") or 3e-4),
            "batch_size": int(v.get("batch_size") or 1),
            "gradient_accumulation_steps": int(v.get("gradient_accumulation_steps") or 16),
            "learning_rate_scheduler": {"warmup_fraction": 0.03},
        },
        "preprocessing": {"global_max_sequence_length": int(v.get("max_sequence_length") or 1024)},
        "backend": {"type": "local"},
    }
    if (v.get("prompt_template") or "").strip():
        # the form stores "\n" typed as two characters; turn them into real newlines
        cfg["prompt"] = {"template": v["prompt_template"].replace("\\n", "\n")}
    if v.get("adapter", "lora") == "lora":
        cfg["adapter"] = {"type": "lora", "r": int(v.get("lora_r") or 8)}
        if v.get("merge_adapter", True):
            cfg["adapter"]["postprocessor"] = {"merge_adapter_into_base_model": True, "progressbar": True}
    bits = str(v.get("quantization_bits") or "none")
    if bits in ("4", "8"):
        cfg["quantization"] = {"bits": int(bits)}
    return cfg


def effective_config(v: Dict[str, Any]) -> Dict[str, Any]:
    """Form-generated config, or the hand-edited YAML when the user switched to raw mode."""
    raw = (v.get("raw_yaml") or "").strip()
    if v.get("use_raw_yaml") and raw:
        cfg = yaml.safe_load(raw)
        if not isinstance(cfg, dict):
            raise ValueError("Ludwig YAML must be a mapping")
        return cfg
    return build_ludwig_config(v)


def to_yaml(cfg: Dict[str, Any]) -> str:
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
