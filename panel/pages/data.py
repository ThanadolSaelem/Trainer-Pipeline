from pathlib import Path

import pandas as pd
import streamlit as st

from core import commands as C
from core import jobs as J
from core import results as R
from core import schema as S
from core import ui
from core.config import save_experiment

settings, exp = ui.page_setup("Data Cleaning", "🧹")
st.caption("ครอบ `Pipeline_clean.py`: sanitize → PDPA → Thai NER → แก้สระลอย → fastText แยก GOOD/BAD")

tab_run, tab_res = st.tabs(["⚙️ ตั้งค่าและรัน", "📊 ผลลัพธ์"])

with tab_run:
    vals = ui.render_form(S.PIPELINE, exp.get("data"), "data")
    exp["data"] = vals
    st.markdown("**Environment**")
    env = settings.env_for("data")
    st.caption(" · ".join(f"`{k}={v}`" for k, v in env.items()) or "ไม่มี (ตั้งได้ที่หน้า Settings)")

    errs = C.validate("data", vals, settings)
    preview_argv = [settings.python_for("data"), "Pipeline_clean.py", *S.build_argv(S.PIPELINE, vals),
                    "--out", "<runs>/data/<job>/output"]
    with st.expander("คำสั่งที่จะรัน", expanded=False):
        ui.show_command(S.redact(preview_argv, S.PIPELINE), str(settings.data_pipeline_dir))

    c1, c2 = st.columns([1, 3])
    run_name = c2.text_input("ชื่อรอบ", value=exp["name"], key="data_run_name")
    if c1.button("▶️ รัน", type="primary", disabled=bool(errs), width="stretch"):
        save_experiment(exp)
        job_dir = J.submit(settings.runs_dir, "data", run_name, C.data_job(vals, settings), {"data": vals})
        ui.submitted("data", job_dir)
    ui.show_errors(errs)
    if vals.get("debug_first_n", 0) == 0:
        st.caption("💡 ลองตั้ง `debug_first_n` = 200 ก่อนรันจริง เพื่อตรวจผลเร็ว ๆ")

with tab_res:
    job = ui.job_picker(settings, "data", "data_job_pick")
    if job:
        ui.job_monitor(job)
        out_dir = Path(job.get("outputs", {}).get("output_dir", ""))
        if out_dir.exists():
            summary = R.pipeline_summary(out_dir)
            stats = summary.get("stats", {})
            outs = R.pipeline_outputs(out_dir)
            st.subheader("สรุป")
            keys = ["processed", "Good", "Bad", "errors_json", "skipped_no_text", "mixed_adjacent"]
            cols = st.columns(len(keys))
            for c, k in zip(cols, keys):
                c.metric(k, f"{stats.get(k, 0):,}")
            if stats.get("processed"):
                bad_ratio = stats.get("Bad", 0) / stats["processed"]
                st.progress(min(1.0, bad_ratio), text=f"สัดส่วน BAD {bad_ratio:.1%}")

            flags_key = (job.get("snapshot", {}).get("data", {}) or {}).get("flags_key") or "__flags"
            fc = R.flag_counts(outs["good"], flags_key) + R.flag_counts(outs["bad"], flags_key)
            if fc:
                st.markdown("**ธงที่ตรวจพบ**")
                st.bar_chart(pd.Series(fc, name="count"))

            st.subheader("ดูตัวอย่างข้อมูล")
            c1, c2, c3 = st.columns([1, 1, 1])
            which = c1.radio("ไฟล์", ["GOOD", "BAD"], horizontal=True, key="data_which")
            mode = c2.radio("วิธีเลือก", ["สุ่ม", "แถวแรก"], horizontal=True, key="data_mode")
            n = c3.number_input("จำนวน", 5, 200, 20, key="data_n")
            path = outs["good"] if which == "GOOD" else outs["bad"]
            if path:
                st.caption(f"{path} · {R.count_lines(path):,} แถว")
                seed = st.session_state.get("data_seed", 0)
                recs = R.sample_jsonl(path, int(n), seed=seed, head=(mode == "แถวแรก"))
                if mode == "สุ่ม" and st.button("🎲 สุ่มใหม่"):
                    st.session_state["data_seed"] = seed + 1
                    st.rerun()
                if recs:
                    st.dataframe(pd.DataFrame(recs), width="stretch", height=420)
                b1, b2 = st.columns(2)
                if b1.button(f"📏 วัด perplexity ของไฟล์ {which}", width="stretch"):
                    exp["perplexity"]["target"] = str(path)
                    save_experiment(exp)
                    st.switch_page("pages/perplexity.py")
                if b2.button(f"🏋️ ใช้ไฟล์ {which} เป็นข้อมูลเทรน", width="stretch"):
                    exp["train"]["dataset"] = str(path)
                    save_experiment(exp)
                    st.switch_page("pages/train.py")
        elif job["status"] == "succeeded":
            st.warning(f"ไม่พบโฟลเดอร์ผลลัพธ์ {out_dir}")
