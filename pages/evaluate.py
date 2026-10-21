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

settings, exp = ui.page_setup("Evaluate", "🎯")
st.caption("ครอบ `evaluation` (`python -m src.run`): ข้อสอบไทย 9 ชุด พร้อมผลราย slice และ confusion matrix")

tab_run, tab_res = st.tabs(["⚙️ ตั้งค่าและรัน", "📊 ผลลัพธ์"])

with tab_run:
    cur = exp.get("eval") or {}
    source = st.session_state.get("f:eval:source", cur.get("source", "hf"))
    hide = {"hf": ("vllm_url", "api_key", "openai_key"),
            "vllm": ("openai_key",),
            "openai": ("vllm_url", "api_key")}.get(source, ())
    vals = ui.render_form(S.EVALUATION, cur, "eval", hide=hide)
    # never persist API keys into the experiment YAML
    exp["eval"] = {k: v for k, v in vals.items() if k not in ("api_key", "openai_key")}

    if vals.get("mode") == "it" and vals.get("thinking_mode") == "True" and not vals.get("do_sample"):
        st.info("thinking mode + greedy อาจวนซ้ำในบางโมเดล ถ้าเจอให้เปิด do_sample")
    errs = C.validate("eval", vals, settings)
    c1, c2 = st.columns([1, 3])
    run_name = c2.text_input("ชื่อรอบ", value=f"{Path(vals.get('model_path') or 'model').name}-{vals.get('mode')}",
                             key="eval_run_name")
    if c1.button("▶️ รัน", type="primary", disabled=bool(errs), width="stretch"):
        save_experiment(exp)
        job_dir = J.submit(settings.runs_dir, "eval", run_name, C.eval_job(vals, settings), {"eval": exp["eval"]})
        ui.submitted("eval", job_dir)
    ui.show_errors(errs)
    st.caption("ข้อมูลข้อสอบจะถูกดาวน์โหลดลง `evaluation/data/` ในการรันครั้งแรก")

with tab_res:
    job = ui.job_picker(settings, "eval", "eval_job_pick")
    if job:
        ui.job_monitor(job)
        res_dir = Path(job.get("outputs", {}).get("results_dir", ""))
        runs = R.find_eval_runs([res_dir])
        if runs:
            run = runs[0]
            reports = R.load_reports(run["dir"])
            st.subheader("คะแนนรวม")
            overall = {ds: rep.get("overall") for ds, rep in reports.items() if isinstance(rep.get("overall"), (int, float))}
            if overall:
                cols = st.columns(min(5, len(overall)))
                for i, (ds, acc) in enumerate(overall.items()):
                    cols[i % len(cols)].metric(ds, f"{acc:.1%}")

            # open on a dataset that has a per-slice breakdown when there is one
            names = list(reports)
            rich = [d for d in names if any(isinstance(v, dict) for v in reports[d].values())]
            ds = st.selectbox("เจาะดูชุดข้อสอบ", names, index=names.index(rich[0]) if rich else 0, key="eval_ds")
            st_tab1, st_tab2, st_tab3 = st.tabs(["ราย slice", "Confusion matrix", "ข้อที่ตอบผิด"])
            preds = R.load_predictions(run["dir"], ds)

            with st_tab1:
                sl = R.slice_table([run], ds)
                if not sl.empty:
                    for t in sl["slice_type"].unique():
                        s = sl[sl["slice_type"] == t].sort_values("accuracy")
                        st.markdown(f"**{t}**")
                        st.altair_chart(
                            alt.Chart(s).mark_bar().encode(
                                x=alt.X("accuracy:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                                y=alt.Y("slice:N", sort="x"), tooltip=["slice", alt.Tooltip("accuracy", format=".1%")]
                            ).properties(height=max(120, 22 * len(s))),
                            width="stretch")
                extra = R.slice_accuracy(preds)
                for col, g in extra.items():
                    if col in set(sl.get("slice_type", [])):
                        continue
                    st.markdown(f"**{col}** (คำนวณจากไฟล์ผลลัพธ์)")
                    st.dataframe(g.style.format({"accuracy": "{:.1%}"}), width="stretch")
                if sl.empty and not extra:
                    st.info("ชุดนี้มีแค่คะแนนรวม")

            with st_tab2:
                cm = R.confusion(preds)
                if cm.empty:
                    st.info("ไม่มีไฟล์ผลลัพธ์รายข้อ")
                else:
                    st.caption("แถว = เฉลย, คอลัมน์ = สิ่งที่โมเดลตอบ · ดูว่าโมเดลเอียงไปตอบตัวเลือกใด "
                               "หรือดึงคำตอบไม่ได้ (∅) บ่อยแค่ไหน")
                    long = cm.reset_index().melt(id_vars="เฉลย", var_name="โมเดลตอบ", value_name="n")
                    base = alt.Chart(long).encode(x="โมเดลตอบ:N", y="เฉลย:N")
                    st.altair_chart(
                        (base.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="blues")))
                         + base.mark_text().encode(text="n:Q")).properties(height=320),
                        width="stretch")
                    pred_dist = preds["predict"].fillna("∅").astype(str).value_counts(normalize=True)
                    st.caption("สัดส่วนคำตอบของโมเดล: " + " · ".join(f"{k} {v:.0%}" for k, v in pred_dist.items()))

            with st_tab3:
                if "correctness" in preds.columns:
                    wrong = preds[preds["correctness"] == 0]
                    st.caption(f"ผิด {len(wrong):,} จาก {len(preds):,} ข้อ")
                    show = [c for c in ("exam_file", "subject", "question", "answer_text", "predict", "response")
                            if c in wrong.columns]
                    st.dataframe(wrong[show], width="stretch", height=420, hide_index=True)
                    st.download_button("⬇️ ดาวน์โหลด (CSV)", wrong.to_csv(index=False).encode("utf-8-sig"),
                                       file_name=f"{job['id']}_{ds}_wrong.csv")
        elif job["status"] == "succeeded":
            st.warning("ไม่พบไฟล์สรุปผล")
