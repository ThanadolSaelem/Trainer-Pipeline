from pathlib import Path

import pandas as pd
import streamlit as st

from core import jobs as J
from core import results as R
from core import ui

settings, exp = ui.page_setup("LLM Control Panel", "🧪")

st.markdown(
    "ที่เดียวสำหรับวงจร **ข้อมูล → วัดความคุ้นเคย → เทรน → ประเมิน → เทียบผล** "
    "ทุกหน้าใช้ค่าจาก experiment ที่เลือกในแถบซ้าย ทุกการรันมีโฟลเดอร์ของตัวเองใน "
    f"`{settings.runs_dir}` พร้อม log และ snapshot ของค่าที่ใช้"
)

STAGES = [
    ("data", "🧹 Data Cleaning", "pages/data.py"),
    ("perplexity", "📏 Perplexity", "pages/perplexity.py"),
    ("train", "🏋️ Train", "pages/train.py"),
    ("eval", "🎯 Evaluate", "pages/evaluate.py"),
]
cols = st.columns(4)
for col, (stage, title, page) in zip(cols, STAGES):
    jobs = J.list_jobs(settings.runs_dir, stage, limit=20)
    last = jobs[0] if jobs else None
    with col.container(border=True):
        st.markdown(f"### {title}")
        if last:
            st.caption(f"ล่าสุด: {ui.STATUS_ICON.get(last['status'], '')} {last['id']}")
        else:
            st.caption("ยังไม่เคยรัน")
        st.caption(f"รันแล้ว {len(jobs)} รอบ")
        if st.button("เปิด", key=f"go:{stage}", width="stretch"):
            st.switch_page(page)

st.subheader("งานล่าสุด")
jobs = J.list_jobs(settings.runs_dir, limit=15)
if jobs:
    st.dataframe(pd.DataFrame([{
        "สถานะ": f"{ui.STATUS_ICON.get(j['status'], '')} {j['status']}",
        "ขั้น": j["stage"], "งาน": j["id"],
        "เริ่ม": (j.get("started") or "").replace("T", " "), "จบ": (j.get("ended") or "").replace("T", " "),
    } for j in jobs]), width="stretch", hide_index=True)
else:
    st.info("ยังไม่มีงาน เริ่มจากหน้า Data Cleaning หรือ Evaluate ได้เลย")

runs = R.find_eval_runs([settings.runs_dir / "eval", *settings.extra_eval_dirs])
if runs:
    st.subheader("ผลประเมินล่าสุด")
    st.dataframe(R.overall_table(runs[:5]).style.format("{:.1%}"), width="stretch")

with st.expander("ตรวจการตั้งค่า"):
    checks = {
        "data-pipeline": (settings.data_pipeline_dir / "Pipeline_clean.py").exists(),
        "evaluation": (settings.evaluation_dir / "src" / "run.py").exists(),
    }
    for name, ok in checks.items():
        st.write(("✅" if ok else "❌") + f" {name}")
    if not all(checks.values()):
        st.caption("แก้ path ได้ที่หน้า Settings")
