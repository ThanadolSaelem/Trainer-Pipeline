"""Declarative parameter schemas: one place that maps form fields to CLI flags.

Each tool's argparse options are described once here. The UI renders forms from
these lists and `build_argv` turns the saved values back into a command line, so
adding a new flag to a tool means adding one `P(...)` line, not new UI code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class P:
    key: str                      # name used in experiment YAML
    flag: Optional[str]           # CLI flag, None = handled by the page itself
    kind: str                     # bool | int | float | str | choice | multi | path | list
    default: Any = None
    help: str = ""
    group: str = "ทั่วไป"
    choices: Sequence[str] = ()
    false_flag: Optional[str] = None   # for bools whose "off" state needs a flag (e.g. --no-resume)
    joiner: Optional[str] = ","        # multi: "," -> one comma-joined value; None -> space-separated values
    advanced: bool = False
    secret: bool = False               # redacted in saved job records and displayed commands
    label: Optional[str] = None

    @property
    def title(self) -> str:
        return self.label or self.key


# ---------------------------------------------------------------- data pipeline
PIPELINE: List[P] = [
    P("input", "--input", "path", "", "โฟลเดอร์หรือไฟล์ JSONL ต้นทาง", "ไฟล์"),
    P("pattern", "--pattern", "str", "*.jsonl", "ลายชื่อไฟล์เมื่อ input เป็นโฟลเดอร์", "ไฟล์"),
    P("text_keys", "--text-keys", "str", "", "คีย์ข้อความ เช่น text (เว้นว่าง = เดาให้)", "ไฟล์"),
    P("combine_keys", "--combine-keys", "str", "", "รวมหลายคีย์เป็นข้อความเดียว เช่น instruction,output", "ไฟล์", advanced=True),

    P("clean_steps", "--clean-steps", "multi", ["html", "emoji", "emoticon"], "ขั้นตอนทำความสะอาดพื้นฐาน", "ทำความสะอาด",
      choices=["html", "bbcode", "emoji", "emoticon", "links", "social", "remove_emails", "remove_phones"]),

    P("pdpa_policy", "--pdpa-policy", "choice", "skip", "วิธีจัดการข้อมูลส่วนบุคคล", "PDPA",
      choices=["skip", "mask", "anonymize", "drop"]),
    P("pdpa_steps", "--pdpa-steps", "multi", [], "ชนิดข้อมูลที่จะจัดการ", "PDPA",
      choices=["name", "email", "phone", "links", "card", "account", "address", "id_card"]),
    P("keep_last", "--keep-last", "int", 0, "จำนวนหลักท้ายที่คงไว้ตอน mask บัตร/บัญชี", "PDPA"),
    P("card_mode", "--card-mode", "choice", "randomize", "", "PDPA", choices=["randomize", "mask", "tokenize"], advanced=True),
    P("account_mode", "--account-mode", "choice", "randomize", "", "PDPA", choices=["randomize", "mask", "tokenize"], advanced=True),
    P("pdpa_salt", "--pdpa-salt", "str", "", "seed ให้ anonymize ได้ผลเดิมทุกครั้ง", "PDPA", advanced=True, secret=True),
    P("addr_gazetteer", "--addr-gazetteer", "path", "", "ไฟล์ gazetteer จังหวัด/อำเภอ/ตำบล (JSON)", "PDPA", advanced=True),
    P("addr_tag", "--addr-tag", "str", "[ADDRESS]", "", "PDPA", advanced=True),
    P("no_addr_keep_province", "--no-addr-keep-province", "bool", False, "ยอมให้สุ่มจังหวัดใหม่ตอน anonymize", "PDPA", advanced=True),

    P("thai_ner_full", "--thai-ner-full", "bool", False, "เปิด Thai NER (ช้า แนะนำ GPU)", "Thai NER"),
    P("thai_ner_policy", "--thai-ner-policy", "choice", "mask", "", "Thai NER", choices=["mask", "anonymize"]),
    P("thai_ner_cats", "--thai-ner-cats", "multi", ["PERSON", "PHONE", "EMAIL", "ADDRESS", "NATIONAL_ID", "HOSPITAL_IDS"],
      "", "Thai NER", choices=["PERSON", "PHONE", "EMAIL", "ADDRESS", "NATIONAL_ID", "HOSPITAL_IDS"]),
    P("thai_ner_map", "--thai-ner-map", "path", "", "ไฟล์แมปโทเค็นสำหรับโหมด anonymize", "Thai NER", advanced=True),
    P("no_public_figure_skip", "--no-public-figure-skip", "bool", False, "ปิดการข้ามชื่อบุคคลสาธารณะ", "Thai NER", advanced=True),

    P("thai_mark_fix", "--thai-mark-fix", "bool", False, "แก้สระลอย/เครื่องหมายซ้ำ", "ภาษาไทย"),
    P("floating_policy", "--floating-policy", "choice", "drop", "", "ภาษาไทย", choices=["drop", "mask", "keep"]),
    P("normalize", "--normalize", "choice", "NFC", "", "ภาษาไทย", choices=["NFC", "NFKC", "none"]),
    P("detect_mixed_script", "--detect-mixed-script", "bool", False, "ตรวจไทยติดกับ Latin/จีน/ญี่ปุ่น", "ภาษาไทย"),
    P("mixed_policy", "--mixed-policy", "choice", "skip", "tag = บันทึกธงใน flags_key", "ภาษาไทย", choices=["skip", "tag"]),
    P("flags_key", "--flags-key", "str", "__flags", "คีย์สำหรับเก็บธงตรวจจับ", "ภาษาไทย"),

    P("fasttext_model", "--fasttext-model", "path", "", "โมเดล fastText แยก GOOD/BAD (เว้นว่าง = ไม่กรอง)", "fastText"),

    P("workers", "--workers", "int", 4, "จำนวน process", "ประสิทธิภาพ"),
    P("chunk_size", "--chunk-size", "int", 50000, "บรรทัดต่อชิ้นงาน", "ประสิทธิภาพ"),
    P("save_every", "--save-every", "int", 5, "บันทึก checkpoint ทุก N ชิ้น", "ประสิทธิภาพ", advanced=True),
    P("single_process", "--single-process", "bool", False, "ไม่ใช้ multiprocessing (ไม่มี resume)", "ประสิทธิภาพ", advanced=True),
    P("gpu_workers_per_gpu", "--gpu-workers-per-gpu", "int", 1, "", "ประสิทธิภาพ", advanced=True),
    P("resume", "--resume", "bool", True, "ทำต่อจาก checkpoint", "ประสิทธิภาพ", false_flag="--no-resume", advanced=True),
    P("clear_ckpt", "--clear-ckpt", "bool", False, "ลบ checkpoint ก่อนเริ่ม", "ประสิทธิภาพ", advanced=True),

    P("debug_first_n", "--debug-first-n", "int", 0, "ลองรันแค่ N แถวแรกต่อไฟล์ (0 = ทั้งหมด)", "Debug"),
    P("debug_show_diff", "--debug-show-diff", "bool", False, "log ข้อความก่อน/หลังทำความสะอาด", "Debug"),
]
PIPELINE_ENV = ["THAI_NER_PATH", "FT_POS_THRESHOLD", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES"]

# ---------------------------------------------------------------- perplexity
PERPLEXITY: List[P] = [
    P("model", "--model", "str", "", "ชื่อโมเดลบน HF หรือ path โลคัล", "โมเดล"),
    P("target", None, "path", "", "ไฟล์หรือโฟลเดอร์ที่จะวัด", "ข้อมูล"),
    P("jsonl_text_field", "--jsonl_text_field", "str", "", "ฟิลด์ข้อความใน JSONL (เว้นว่าง = เดาให้)", "ข้อมูล"),
    P("csv_text_col", "--csv_text_col", "str", "", "คอลัมน์ข้อความใน CSV", "ข้อมูล", advanced=True),
    P("max_rows", "--max_rows", "int", 0, "จำกัดจำนวนแถวต่อไฟล์ (0 = ทั้งหมด)", "ข้อมูล"),
    P("context_length", "--context_length", "int", 2048, "ความยาวหน้าต่างโทเค็น", "การวัด"),
    P("overlap_ratio", "--overlap_ratio", "float", 0.25, "สัดส่วน overlap ระหว่างหน้าต่าง", "การวัด"),
    P("batch_size", "--batch_size", "int", 4, "", "การวัด"),
    P("use_chat_template", "--use_chat_template", "choice", "none", "none = เดาจากชื่อโมเดล", "การวัด",
      choices=["none", "true", "false"]),
    P("top_k_tokens", "--top_k_tokens", "int", 5, "เก็บโทเค็นที่ loss สูงสุด k ตัวต่อข้อความ", "การวิเคราะห์"),
    P("md_handling", "--md_handling", "choice", "auto", "", "การวัด", choices=["auto", "force", "off"], advanced=True),
    P("md_strip_code_blocks", "--md_strip_code_blocks", "choice", "true", "", "การวัด", choices=["true", "false"], advanced=True),
]

# ---------------------------------------------------------------- evaluation
EVAL_DATASETS = ["belebele", "xcopa", "mmlu_thai", "xnli", "m3exam", "thai_exam", "m6exam", "mmlu", "mmlu_proX_thai"]
EVALUATION: List[P] = [
    P("model_path", "--model-path", "str", "", "ชื่อโมเดล / path checkpoint / ชื่อโมเดลบน API", "โมเดล"),
    P("source", None, "choice", "hf", "hf = โหลดโมเดลบนเครื่อง", "โมเดล", choices=["hf", "vllm", "openai"]),
    P("vllm_url", "--vllm", "str", "", "Base URL ของ vLLM server", "โมเดล"),
    P("api_key", "--api-key", "str", "", "API key ของ vLLM (ถ้ามี)", "โมเดล", secret=True),
    P("openai_key", "--openai", "str", "", "OpenAI API key (เว้นว่าง = ใช้ env OPENAI_API_KEY)", "โมเดล", secret=True),
    P("mode", "--mode", "choice", "pt", "pt = base model (logit), it = chat model (generate)", "การประเมิน",
      choices=["pt", "it"]),
    P("thinking_mode", "--thinking-mode", "choice", "None", "", "การประเมิน",
      choices=["None", "True", "False", "no_chat_template"]),
    P("data", "--data", "multi", list(EVAL_DATASETS), "ชุดข้อสอบ", "การประเมิน", choices=EVAL_DATASETS, joiner=None),
    P("num_shots", "--num-shots", "int", 5, "", "การประเมิน"),
    P("batch_size", "--batch-size", "int", 4, "", "การประเมิน"),
    P("max_seq_len", "--max-seq-len", "int", 8192, "", "การประเมิน", advanced=True),
    P("seed", "--seed", "int", 47, "", "การประเมิน", advanced=True),
    P("do_sample", "--do-sample", "bool", False, "ใช้ sampling (ค่าเริ่มต้น greedy เพื่อผลซ้ำได้)", "การประเมิน", advanced=True),
    P("debug", "--debug", "bool", False, "ทดลองแค่ 5 ข้อแรกต่อชุด", "การประเมิน"),
]

# ---------------------------------------------------------------- ludwig (LLM fine-tune)
LUDWIG: List[P] = [
    P("base_model", None, "str", "", "โมเดลตั้งต้น (HF name หรือ path)", "โมเดล"),
    P("dataset", None, "path", "", "ไฟล์ข้อมูลเทรน (jsonl / csv / parquet)", "ข้อมูล"),
    P("input_column", None, "str", "instruction", "คอลัมน์ input", "ข้อมูล"),
    P("output_column", None, "str", "output", "คอลัมน์ output", "ข้อมูล"),
    P("prompt_template", None, "str", "", "template เช่น ### คำสั่ง:\\n{instruction}\\n### คำตอบ: (เว้นว่าง = ใช้ input ตรง ๆ)", "ข้อมูล"),
    P("adapter", None, "choice", "lora", "", "การเทรน", choices=["lora", "none"]),
    P("lora_r", None, "int", 8, "", "การเทรน"),
    P("quantization_bits", None, "choice", "none", "4 = QLoRA (ต้องมี GPU + bitsandbytes)", "การเทรน",
      choices=["none", "4", "8"]),
    P("epochs", None, "int", 3, "", "การเทรน"),
    P("learning_rate", None, "float", 0.0003, "", "การเทรน"),
    P("batch_size", None, "int", 1, "", "การเทรน"),
    P("gradient_accumulation_steps", None, "int", 16, "", "การเทรน"),
    P("max_sequence_length", None, "int", 1024, "", "การเทรน", advanced=True),
    P("merge_adapter", None, "bool", True, "รวม LoRA เข้าโมเดลหลักตอนบันทึก (นำไปประเมินต่อได้ทันที)", "การเทรน"),
]


SCHEMAS: Dict[str, List[P]] = {
    "data": PIPELINE,
    "perplexity": PERPLEXITY,
    "eval": EVALUATION,
    "train": LUDWIG,
}


def defaults(schema: List[P]) -> Dict[str, Any]:
    return {p.key: (list(p.default) if isinstance(p.default, list) else p.default) for p in schema}


def merged(schema: List[P], values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = defaults(schema)
    for k, v in (values or {}).items():
        if k in out:
            out[k] = v
    return out


def _empty(v: Any) -> bool:
    return v is None or v == "" or v == []


def build_argv(schema: List[P], values: Dict[str, Any], skip: Sequence[str] = ()) -> List[str]:
    """Turn form values into CLI arguments. Empty values and unset bools are omitted."""
    argv: List[str] = []
    vals = merged(schema, values)
    for p in schema:
        if p.flag is None or p.key in skip:
            continue
        v = vals.get(p.key)
        if p.kind == "bool":
            if v:
                if not p.false_flag:
                    argv.append(p.flag)
            elif p.false_flag:
                argv.append(p.false_flag)
            continue
        if _empty(v):
            continue
        if p.kind in ("int", "float") and v == 0 and p.default == 0:
            continue  # 0 means "off" for these options
        if p.kind == "multi":
            items = [str(x) for x in v]
            if p.joiner is None:
                argv += [p.flag, *items]
            else:
                argv += [p.flag, p.joiner.join(items)]
            continue
        argv += [p.flag, str(v)]
    return argv


def redact(argv: List[str], schema: List[P]) -> List[str]:
    """Replace values of secret flags with *** for logs and saved records."""
    secret_flags = {p.flag for p in schema if p.secret and p.flag}
    out: List[str] = []
    hide_next = False
    for a in argv:
        if hide_next:
            out.append("***")
            hide_next = False
            continue
        out.append(a)
        if a in secret_flags:
            hide_next = True
    return out
