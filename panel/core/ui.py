"""Shared Streamlit pieces: sidebar experiment picker, schema forms, job monitor."""
from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from . import jobs as J
from . import schema as S
from .config import (Settings, list_experiments, load_experiment, load_settings, save_experiment,
                     slugify)

STATUS_ICON = {
    "queued": "⏳", "running": "🔄", "succeeded": "✅", "failed": "❌", "cancelled": "⛔", "lost": "❓",
}


def page_setup(title: str, icon: str = "🧪") -> tuple[Settings, Dict[str, Any]]:
    settings = load_settings()
    exp = experiment_sidebar()
    st.title(f"{icon} {title}")
    return settings, exp


def experiment_sidebar() -> Dict[str, Any]:
    names = list_experiments()
    if "exp" not in st.session_state:
        st.session_state.exp = load_experiment(names[0] if names else "default")
    exp = st.session_state.exp

    with st.sidebar:
        st.markdown("### Experiment")
        options = names if exp["name"] in names else [exp["name"], *names]
        choice = st.selectbox("ชุดการตั้งค่า", options, index=options.index(exp["name"]), key="exp_select",
                              help="ทุกหน้าจะใช้ค่าจาก experiment นี้ และบันทึกกลับลงไฟล์ experiments/<ชื่อ>.yaml")
        if choice != exp["name"]:
            st.session_state.exp = load_experiment(choice)
            _clear_form_state()
            st.rerun()
        with st.popover("➕ สร้าง / คัดลอก", width="stretch"):
            new_name = st.text_input("ชื่อใหม่", key="new_exp_name")
            copy = st.checkbox("คัดลอกค่าจาก experiment ปัจจุบัน", value=True)
            if st.button("สร้าง", disabled=not new_name):
                base = dict(exp) if copy else load_experiment("__blank__")
                base = {**base, "name": slugify(new_name)}
                save_experiment(base)
                st.session_state.exp = load_experiment(base["name"])
                _clear_form_state()
                st.rerun()
        exp["notes"] = st.text_area("โน้ต", exp.get("notes", ""), height=80, key="exp_notes")
        if st.button("💾 บันทึก experiment", width="stretch"):
            save_experiment(exp)
            st.toast(f"บันทึก experiments/{exp['name']}.yaml แล้ว")
        running = [j for j in J.list_jobs(load_settings().runs_dir, limit=50) if j["status"] in J.ACTIVE]
        if running:
            st.markdown("---")
            st.markdown(f"**กำลังรัน {len(running)} งาน**")
            for j in running[:5]:
                st.caption(f"🔄 {j['stage']} · {j['name']}")
    return exp


def _clear_form_state() -> None:
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith("f:"):
            del st.session_state[k]


# ------------------------------------------------------------------ forms
def _widget(p: S.P, value: Any, key: str) -> Any:
    label = p.title
    help_ = p.help or None
    if p.kind == "bool":
        return st.toggle(label, value=bool(value), key=key, help=help_)
    if p.kind == "int":
        return int(st.number_input(label, value=int(value or 0), step=1, key=key, help=help_))
    if p.kind == "float":
        return float(st.number_input(label, value=float(value or 0.0), format="%.6g", key=key, help=help_))
    if p.kind == "choice":
        opts = list(p.choices)
        idx = opts.index(str(value)) if str(value) in opts else 0
        return st.selectbox(label, opts, index=idx, key=key, help=help_)
    if p.kind == "multi":
        opts = list(p.choices)
        cur = [x for x in (value or []) if x in opts]
        return st.multiselect(label, opts, default=cur, key=key, help=help_)
    return st.text_input(label, value=str(value or ""), key=key, help=help_,
                         type="password" if p.secret else "default")


def render_form(schema: List[S.P], values: Dict[str, Any], stage: str,
                hide: tuple = (), columns: int = 2) -> Dict[str, Any]:
    """Grouped form. Returns the full value dict (defaults filled in)."""
    vals = S.merged(schema, values)
    groups: Dict[str, List[S.P]] = {}
    for p in schema:
        if p.key not in hide:
            groups.setdefault(p.group, []).append(p)

    for group, params in groups.items():
        basic = [p for p in params if not p.advanced]
        adv = [p for p in params if p.advanced]
        st.markdown(f"**{group}**")
        cols = st.columns(columns)
        for i, p in enumerate(basic):
            with cols[i % columns]:
                vals[p.key] = _widget(p, vals.get(p.key), f"f:{stage}:{p.key}")
        if adv:
            with st.expander(f"ขั้นสูง: {group}"):
                acols = st.columns(columns)
                for i, p in enumerate(adv):
                    with acols[i % columns]:
                        vals[p.key] = _widget(p, vals.get(p.key), f"f:{stage}:{p.key}")
    return vals


def show_command(argv: List[str], cwd: str) -> None:
    st.code(f"cd {shlex.quote(cwd)}\n" + " \\\n  ".join(shlex.quote(a) for a in argv), language="bash")


def show_errors(errs: List[str]) -> bool:
    for e in errs:
        st.warning(e)
    return not errs


# ------------------------------------------------------------------ jobs
# log patterns that mean "it ran, but not the way you think"
LOG_HINTS = [
    (re.compile(r"\[import\] f_text_cleaner not available"),
     "โหลด f_text_cleaner ไม่ได้: ขั้น sanitize / PDPA / Thai NER ไม่ได้ทำงานจริง ข้อความผ่านไปเหมือนเดิม "
     "ตรวจ dependency (bs4, Thai-Data-Privacy, corpus ของ pythainlp) ใน python ของขั้น data"),
    (re.compile(r"\[fastText\] disabled"), "โหลดโมเดล fastText ไม่ได้: ทุกแถวจะถูกจัดเป็น GOOD"),
    (re.compile(r"No records were processed"), "ไม่มีแถวถูกประมวลผล: ตรวจ text_keys หรือรูปแบบไฟล์ input"),
    (re.compile(r"\[clean-step\] unknown step ignored: '([^']+)'"), "ขั้นทำความสะอาดที่ไม่รู้จักถูกข้าม: {0}"),
    (re.compile(r"(?:ModuleNotFoundError|ImportError): No module named '([^']+)'"), "ขาด module '{0}' ใน python ของขั้นนี้ (ตั้งค่า/ติดตั้งที่หน้า Settings)"),
    (re.compile(r"CUDA out of memory|MPS backend out of memory"), "หน่วยความจำ GPU ไม่พอ: ลด batch size หรือ context length"),
    (re.compile(r"gazetteer not found"), "ไม่พบไฟล์ gazetteer ที่อยู่: การตรวจที่อยู่ใช้ regex อย่างเดียว"),
]


def log_hints(log: str) -> List[str]:
    out = []
    for rx, msg in LOG_HINTS:
        m = rx.search(log)
        if m:
            out.append(msg.format(*m.groups()) if m.groups() else msg)
    return out


def job_label(j: Dict[str, Any]) -> str:
    return f"{STATUS_ICON.get(j['status'], '•')} {j['id']}"


def job_picker(settings: Settings, stage: str, key: str, only_ok: bool = False,
               label: str = "เลือกรอบที่รัน") -> Optional[Dict[str, Any]]:
    jobs = J.list_jobs(settings.runs_dir, stage)
    if only_ok:
        jobs = [j for j in jobs if j["status"] == "succeeded"]
    if not jobs:
        st.info("ยังไม่มีรอบที่รันในขั้นนี้")
        return None
    ids = [j["id"] for j in jobs]
    default = st.session_state.get(f"last_job:{stage}")
    idx = ids.index(default) if default in ids else 0
    pick = st.selectbox(label, ids, index=idx, key=key,
                        format_func=lambda i: job_label(next(j for j in jobs if j["id"] == i)))
    return next(j for j in jobs if j["id"] == pick)


def job_monitor(job: Dict[str, Any]) -> None:
    """Status, step list and live log (refreshes itself while the job runs)."""
    job_dir = Path(job["dir"])

    @st.fragment(run_every=2 if job["status"] in J.ACTIVE else None)
    def _live():
        j = J.read_job(job_dir) or job
        status = j["status"]
        started, ended = j.get("started"), j.get("ended")
        dur = ""
        if started:
            from datetime import datetime
            t1 = datetime.fromisoformat(ended) if ended else datetime.now()
            secs = int((t1 - datetime.fromisoformat(started)).total_seconds())
            dur = f"{secs // 3600}:{secs % 3600 // 60:02d}:{secs % 60:02d}" if secs >= 3600 else f"{secs // 60}:{secs % 60:02d}"
        steps = " → ".join(
            ("✅ " if s.get("returncode") == 0 else "❌ " if s.get("returncode") not in (None, 0) else "… ") + s["title"]
            for s in j.get("steps", []))
        c1, c2 = st.columns([6, 1])
        c1.markdown(f"#### {STATUS_ICON.get(status, '')} {status}"
                    + (f" <span style='opacity:.6;font-size:.8em'>· เริ่ม {started.replace('T', ' ')} · {'ใช้เวลา' if ended else 'ผ่านไป'} {dur}</span>"
                       if started else ""), unsafe_allow_html=True)
        c1.caption(steps)
        if status in J.ACTIVE:
            if c2.button("⛔ หยุด", key=f"cancel:{j['id']}", width="stretch"):
                J.cancel(job_dir)
                st.toast("ส่งคำสั่งหยุดแล้ว")
        for hint in log_hints(J.tail(job_dir / "log.txt", 400_000)):
            st.warning(hint, icon="⚠️")
        log = J.tail(job_dir / "log.txt")
        with st.expander("Log", expanded=status != "succeeded"):
            st.code(log[-20000:] or "(ยังไม่มี log)", language="text", height=360)
        if status not in J.ACTIVE and job["status"] in J.ACTIVE:
            st.rerun()  # finished: refresh the whole page so results appear

    _live()
    with st.expander("รายละเอียดงาน (job.json)"):
        st.json({k: v for k, v in job.items() if k != "dir"}, expanded=False)
        st.caption(f"โฟลเดอร์: {job_dir}")


def submitted(stage: str, job_dir: Path) -> None:
    st.session_state[f"last_job:{stage}"] = job_dir.name
    st.toast(f"เริ่มงาน {job_dir.name}")
    st.rerun()
