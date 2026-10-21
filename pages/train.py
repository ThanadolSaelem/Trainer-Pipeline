import json
import subprocess
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from core import commands as C
from core import jobs as J
from core import results as R
from core import schema as S
from core import ui
from core.config import save_experiment
from core.ludwig_cfg import build_ludwig_config, to_yaml

settings, exp = ui.page_setup("Train (Ludwig)", "🏋️")
st.caption("สร้าง config ของ Ludwig จากฟอร์ม (LLM fine-tune + LoRA) แล้วรัน `ludwig train` "
           "ตามด้วย export เป็นโฟลเดอร์ Hugging Face ที่หน้า Evaluate โหลดได้ทันที")


@st.cache_data(ttl=60, show_spinner=False)
def ludwig_version(py: str) -> str:
    try:
        r = subprocess.run([py, "-c", "import ludwig; print(ludwig.__version__)"],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def peek_columns(path: str):
    p = Path(path).expanduser()
    if not p.exists():
        return None
    try:
        if p.suffix in (".jsonl", ".ndjson"):
            with open(p, encoding="utf-8") as f:
                rows = [json.loads(next(f)) for _ in range(3)]
            return pd.DataFrame(rows)
        if p.suffix == ".csv":
            return pd.read_csv(p, nrows=3)
        if p.suffix == ".parquet":
            return pd.read_parquet(p).head(3)
    except StopIteration:
        return None
    except Exception:
        return None
    return None


tab_run, tab_res = st.tabs(["⚙️ ตั้งค่าและรัน", "📈 ผลการเทรน"])

with tab_run:
    ver = ludwig_version(settings.python_for("train"))
    if ver:
        st.success(f"พบ Ludwig {ver} ใน `{settings.python_for('train')}`")
    else:
        st.warning("ยังไม่พบ Ludwig ใน Python ของขั้น train ติดตั้งด้วย `pip install \"ludwig[llm]\"` "
                   "(ต้อง Python 3.12+) หรือชี้ไปยัง env ที่มี Ludwig ในหน้า Settings")

    cur = exp.get("train") or {}
    vals = ui.render_form(S.LUDWIG, cur, "train")

    cols_df = peek_columns(vals.get("dataset") or "")
    if cols_df is not None:
        missing = [c for c in (vals["input_column"], vals["output_column"]) if c not in cols_df.columns]
        st.caption("คอลัมน์ในไฟล์: " + ", ".join(f"`{c}`" for c in cols_df.columns))
        if missing:
            st.warning(f"ไม่พบคอลัมน์ {missing} ในไฟล์ข้อมูล (ข้อมูลเทรน LLM ต้องมีคู่ input/output; "
                       "ไฟล์ที่ผ่าน Data Cleaning ซึ่งมีแต่ `text` ต้องแปลงเป็นคู่คำถาม-คำตอบก่อน)")

    st.markdown("**Ludwig config**")
    use_raw = st.toggle("แก้ YAML เอง", value=bool(cur.get("use_raw_yaml")), key="train_use_raw",
                        help="ปิด = สร้างจากฟอร์มด้านบน, เปิด = ใช้ YAML ที่แก้เอง (ฟอร์มจะไม่มีผล)")
    generated = to_yaml(build_ludwig_config(vals)) if vals.get("base_model") else ""
    if use_raw:
        raw = st.text_area("YAML", value=cur.get("raw_yaml") or generated, height=360, key="train_raw_yaml")
    else:
        st.code(generated or "# กรอก base_model เพื่อดู config", language="yaml")
        raw = cur.get("raw_yaml", "")
    vals["use_raw_yaml"] = use_raw
    vals["raw_yaml"] = raw
    exp["train"] = vals

    errs = C.validate("train", vals, settings)
    c1, c2 = st.columns([1, 3])
    run_name = c2.text_input("ชื่อรอบ", value=exp["name"], key="train_run_name")
    if c1.button("▶️ เริ่มเทรน", type="primary", disabled=bool(errs) or not ver, width="stretch"):
        save_experiment(exp)
        job_dir = J.submit(settings.runs_dir, "train", run_name,
                           C.train_job(vals, settings, experiment_name=exp["name"]), {"train": vals})
        ui.submitted("train", job_dir)
    ui.show_errors(errs)
    st.caption("💡 บน Mac (Apple Silicon) ใช้ quantization = none: bitsandbytes 4/8-bit ต้องใช้ GPU NVIDIA")

with tab_res:
    job = ui.job_picker(settings, "train", "train_job_pick")
    if job:
        ui.job_monitor(job)
        outs = job.get("outputs", {})
        stats = R.ludwig_stats(Path(outs.get("ludwig_dir", "")))
        if not stats.empty:
            st.subheader("Learning curves")
            metric = st.selectbox("metric", sorted(stats["metric"].unique()),
                                  index=sorted(stats["metric"].unique()).index("loss")
                                  if "loss" in set(stats["metric"]) else 0, key="train_metric")
            s = stats[stats["metric"] == metric]
            st.altair_chart(
                alt.Chart(s).mark_line(point=True).encode(
                    x="epoch:Q", y=alt.Y("value:Q", title=metric, scale=alt.Scale(zero=False)), color="split:N",
                    strokeDash="feature:N", tooltip=["split", "feature", "epoch", "value"]),
                width="stretch")
        hf = Path(outs.get("hf_model", ""))
        if (hf / "config.json").exists():
            st.success(f"โมเดลพร้อมประเมิน: `{hf}`")
            if st.button("🎯 ประเมินโมเดลนี้", type="primary"):
                exp["eval"]["model_path"] = str(hf)
                exp["eval"]["source"] = "hf"
                save_experiment(exp)
                st.switch_page("pages/evaluate.py")
        if outs.get("config"):
            with st.expander("config ที่ใช้"):
                p = Path(outs["config"])
                st.code(p.read_text(encoding="utf-8") if p.exists() else "", language="yaml")
