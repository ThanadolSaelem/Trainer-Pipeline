from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import commands as C
from core import jobs as J
from core import results as R
from core import schema as S
from core import ui
from core.config import save_experiment

settings, exp = ui.page_setup("Perplexity", "📏")
st.caption("ครอบ `Perplexity.py`: วัดความคุ้นเคยของโมเดลกับข้อความ รายไฟล์และรายข้อความ "
           "พร้อม chars/token และโทเค็นที่โมเดลสะดุด")


def recent_data_outputs():
    out = []
    for j in J.list_jobs(settings.runs_dir, "data", limit=30):
        d = Path(j.get("outputs", {}).get("output_dir", ""))
        if d.exists():
            out += [str(p) for p in sorted(d.glob("*.jsonl"))]
    return out


tab_run, tab_res = st.tabs(["⚙️ ตั้งค่าและรัน", "📊 ผลลัพธ์"])

with tab_run:
    cur = exp.get("perplexity") or {}
    recent = recent_data_outputs()
    if recent:
        pick = st.selectbox("เลือกจากผลลัพธ์ Data Cleaning ล่าสุด (ไม่บังคับ)", ["—", *recent], key="ppl_recent")
        if pick != "—" and pick != cur.get("target"):
            cur["target"] = pick
            st.session_state["f:perplexity:target"] = pick
    vals = ui.render_form(S.PERPLEXITY, cur, "perplexity")
    exp["perplexity"] = vals

    errs = C.validate("perplexity", vals, settings)
    with st.expander("คำสั่งที่จะรัน"):
        mode = ["folder", "--folder"] if vals.get("target") and Path(vals["target"]).is_dir() else ["--file"]
        ui.show_command([settings.python_for("perplexity"), "Perplexity.py", *mode, vals.get("target", ""),
                         *S.build_argv(S.PERPLEXITY, vals), "--output_json", "<job>/summary.json",
                         "--per_text_jsonl", "<job>/per_text.jsonl"], str(settings.data_pipeline_dir))
    c1, c2 = st.columns([1, 3])
    run_name = c2.text_input("ชื่อรอบ", value=f"{exp['name']}-{Path(vals.get('model') or 'model').name}",
                             key="ppl_run_name")
    if c1.button("▶️ รัน", type="primary", disabled=bool(errs), width="stretch"):
        save_experiment(exp)
        job_dir = J.submit(settings.runs_dir, "perplexity", run_name, C.perplexity_job(vals, settings),
                           {"perplexity": vals})
        ui.submitted("perplexity", job_dir)
    ui.show_errors(errs)

with tab_res:
    job = ui.job_picker(settings, "perplexity", "ppl_job_pick")
    if job:
        ui.job_monitor(job)
        job_dir = Path(job["dir"])
        summ = R.ppl_summary(job_dir)
        df = R.ppl_per_text(job_dir)
        if summ:
            st.subheader("สรุป")
            c = st.columns(4)
            ds = summ.get("PPL_dataset", summ.get("PPL_micro"))
            c[0].metric("PPL (token-level)", f"{ds:.3f}" if isinstance(ds, (int, float)) else "-",
                        help="exp(NLL รวม / จำนวนโทเค็นรวม)")
            mac = summ.get("PPL_macro_folder", summ.get("PPL_macro"))
            c[1].metric("PPL macro", f"{mac:.3f}" if isinstance(mac, (int, float)) else "-",
                        help="ค่าเฉลี่ย PPL รายไฟล์/รายข้อความ (โดนค่าโด่งดึงได้ง่าย)")
            c[2].metric("โทเค็นที่วัด", f"{summ.get('total_tokens', summ.get('tokens_evaluated', 0)):,}")
            c[3].metric("ข้อความ", f"{len(df):,}")
        if not df.empty:
            d = df.dropna(subset=["ppl"]).copy()
            d["log10_ppl"] = np.log10(d["ppl"].clip(lower=1e-9))
            st.subheader("การกระจายของ PPL รายข้อความ")
            st.altair_chart(
                alt.Chart(d).mark_bar().encode(
                    alt.X("log10_ppl:Q", bin=alt.Bin(maxbins=40), title="log10(PPL)"),
                    alt.Y("count()", title="จำนวนข้อความ")).properties(height=220),
                width="stretch")

            st.subheader("แยกปัญหา vocab ออกจากปัญหาเนื้อหา")
            st.caption("แกน X = ตัวอักษรต่อโทเค็น (ต่ำ = tokenizer หั่นละเอียด มักเป็นปัญหา vocab/ภาษา) · "
                       "แกน Y = PPL · มุมซ้ายบน: tokenizer ไม่รู้จักและโมเดลไม่คุ้น / มุมขวาบน: ตัดคำได้ดีแต่เนื้อหาแปลก "
                       "(ข้อมูลขยะ, โดเมนใหม่, label ผิด)")
            if "chars_per_token" in d.columns:
                st.altair_chart(
                    alt.Chart(d).mark_circle(opacity=0.6).encode(
                        alt.X("chars_per_token:Q", title="chars / token"),
                        alt.Y("ppl:Q", scale=alt.Scale(type="log"), title="PPL (log)"),
                        alt.Color("file:N", legend=alt.Legend(orient="bottom")),
                        tooltip=["file", "index", "ppl", "tokens", "chars_per_token", "worst", "worst_context"],
                    ).properties(height=360).interactive(),
                    width="stretch")

            st.subheader("ข้อความที่ PPL สูง/ต่ำผิดปกติ")
            lo_q, hi_q = st.slider("ช่วงเปอร์เซ็นไทล์ที่ถือว่าปกติ", 0, 100, (2, 98), key="ppl_q")
            lo, hi = np.percentile(d["ppl"], [lo_q, hi_q])
            hi_df = d[d["ppl"] > hi].sort_values("ppl", ascending=False)
            lo_df = d[d["ppl"] < lo].sort_values("ppl")
            cols = ["file", "index", "ppl", "tokens", "chars_per_token", "worst", "worst_context"]
            a, b = st.columns(2)
            a.markdown(f"**สูงผิดปกติ** (> {hi:.2f}) · {len(hi_df)} ข้อความ")
            a.caption("มักเป็นข้อมูลขยะ OCR พัง ภาษาอื่นปน หรือโดเมนที่โมเดลไม่เคยเห็น")
            a.dataframe(hi_df[cols], width="stretch", height=280, hide_index=True)
            b.markdown(f"**ต่ำผิดปกติ** (< {lo:.2f}) · {len(lo_df)} ข้อความ")
            b.caption("มักเป็นข้อความซ้ำ/boilerplate หรือข้อมูลที่อาจหลุดมาจากชุดทดสอบ")
            b.dataframe(lo_df[cols], width="stretch", height=280, hide_index=True)
            st.download_button("⬇️ ดาวน์โหลดรายการผิดปกติ (CSV)",
                               pd.concat([hi_df.assign(kind="high"), lo_df.assign(kind="low")])[cols + ["kind"]]
                               .to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{job['id']}_outliers.csv")

            target = job.get("outputs", {}).get("target")
            if target:
                with st.expander("🔍 เปิดดูข้อความจริง"):
                    c1, c2 = st.columns(2)
                    f = c1.selectbox("ไฟล์", sorted(d["file"].unique()), key="ppl_view_file")
                    idx = c2.number_input("index", 0, int(d["index"].max()), int(hi_df["index"].iloc[0]) if len(hi_df) else 0,
                                          key="ppl_view_idx")
                    field = (job.get("snapshot", {}).get("perplexity", {}) or {}).get("jsonl_text_field") or None
                    text = R.text_lookup(Path(target), f, int(idx), field)
                    row = d[(d["file"] == f) & (d["index"] == idx)]
                    if not row.empty:
                        st.caption(f"PPL {row['ppl'].iloc[0]:.2f} · worst: {row['worst'].iloc[0]}")
                    st.text(text[:5000] if text else "(อ่านข้อความต้นฉบับไม่ได้: รองรับเฉพาะ jsonl หรือไฟล์ข้อความเดี่ยว)")

            wt = R.worst_token_table(df)
            if not wt.empty:
                st.subheader("โทเค็นที่โมเดลสะดุดบ่อย")
                st.caption("โทเค็นที่ติด top-loss ของหลายข้อความ: ถ้าเป็นคำไทยถูกหั่นเป็นเศษ แสดงว่าเป็นปัญหา tokenizer; "
                           "ถ้าเป็นคำเต็มเฉพาะทาง แสดงว่าโมเดลขาดความรู้โดเมน")
                st.dataframe(wt, width="stretch", hide_index=True)
