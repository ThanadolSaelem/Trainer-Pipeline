from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from core import results as R
from core import ui

settings, exp = ui.page_setup("Compare models", "📊")
st.caption("เทียบผลประเมินหลายรอบ/หลายโมเดล ทั้งคะแนนรวมและราย slice")

roots = [settings.runs_dir / "eval", *settings.extra_eval_dirs]
runs = R.find_eval_runs(roots)
if not runs:
    st.info("ยังไม่มีผลประเมิน รันที่หน้า Evaluate หรือเพิ่มโฟลเดอร์ results เดิมในหน้า Settings")
    st.stop()

labels = [r["label"] for r in runs]
picked = st.multiselect("เลือกรอบที่จะเทียบ", labels, default=labels[: min(3, len(labels))])
sel = [r for r in runs if r["label"] in picked]
if not sel:
    st.stop()

table = R.overall_table(sel)
st.subheader("คะแนนรวม")
st.dataframe(table.style.format("{:.1%}").background_gradient(axis=0, cmap="Greens"), width="stretch")

long = table.drop(columns=["mean"], errors="ignore").reset_index().melt(id_vars="run", var_name="dataset",
                                                                        value_name="accuracy").dropna()
st.altair_chart(
    alt.Chart(long).mark_bar().encode(
        x=alt.X("run:N", title=None, axis=alt.Axis(labels=False)), y=alt.Y("accuracy:Q", axis=alt.Axis(format="%")),
        color="run:N", column=alt.Column("dataset:N", title=None),
        tooltip=["run", "dataset", alt.Tooltip("accuracy", format=".1%")]).properties(width=90, height=220))

if len(sel) >= 2:
    st.subheader("ส่วนต่างเทียบกับรอบอ้างอิง")
    base = st.selectbox("รอบอ้างอิง", list(table.index), key="cmp_base")
    delta = (table - table.loc[base]).drop(index=base)
    st.dataframe(delta.style.format("{:+.1%}").background_gradient(axis=None, cmap="RdYlGn", vmin=-0.1, vmax=0.1),
                 width="stretch")

st.subheader("ราย slice")
datasets = sorted({ds for r in sel for ds in R.load_reports(r["dir"])})
ds = st.selectbox("ชุดข้อสอบ", datasets, key="cmp_ds")
sl = R.slice_table(sel, ds)
if sl.empty:
    st.info("ชุดนี้มีแค่คะแนนรวม")
else:
    for t in sl["slice_type"].unique():
        s = sl[sl["slice_type"] == t]
        st.markdown(f"**{t}**")
        piv = s.pivot_table(index="slice", columns="run", values="accuracy")
        st.dataframe(piv.style.format("{:.1%}").background_gradient(axis=1, cmap="Greens"), width="stretch")
