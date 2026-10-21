import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from core import jobs as J
from core import ui

settings, exp = ui.page_setup("All jobs", "🗂️")

stage = st.segmented_control("ขั้น", ["ทั้งหมด", *J.STAGES], default="ทั้งหมด", key="jobs_stage")
jobs = J.list_jobs(settings.runs_dir, None if stage in (None, "ทั้งหมด") else stage)
if not jobs:
    st.info("ยังไม่มีงาน")
    st.stop()

st.dataframe(pd.DataFrame([{
    "สถานะ": f"{ui.STATUS_ICON.get(j['status'], '')} {j['status']}", "ขั้น": j["stage"], "งาน": j["id"],
    "เริ่ม": (j.get("started") or "").replace("T", " "), "จบ": (j.get("ended") or "").replace("T", " "),
} for j in jobs]), width="stretch", hide_index=True, height=280)

ids = [j["id"] for j in jobs]
pick = st.selectbox("เปิดดูงาน", ids, format_func=lambda i: ui.job_label(next(j for j in jobs if j["id"] == i)))
job = next(j for j in jobs if j["id"] == pick)
ui.job_monitor(job)

if job["status"] not in J.ACTIVE:
    with st.popover("🗑️ ลบงานนี้"):
        st.write(f"ลบโฟลเดอร์ `{job['dir']}` ทั้งหมด (รวมผลลัพธ์) กู้คืนไม่ได้")
        if st.button("ยืนยันลบ", type="primary"):
            shutil.rmtree(job["dir"], ignore_errors=True)
            st.rerun()
