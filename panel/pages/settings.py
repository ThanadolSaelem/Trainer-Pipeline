import subprocess
import sys

import streamlit as st
import yaml

from core import ui
from core.config import DEFAULT_SETTINGS, SETTINGS_FILE, load_settings

settings, exp = ui.page_setup("Settings", "⚙️")
raw = settings.raw

st.subheader("ที่อยู่ของโปรเจกต์")
c1, c2 = st.columns(2)
raw["data_pipeline_dir"] = c1.text_input("data-pipeline", raw["data_pipeline_dir"])
raw["evaluation_dir"] = c2.text_input("evaluation", raw["evaluation_dir"])
raw["runs_dir"] = c1.text_input("โฟลเดอร์เก็บผลการรัน", raw["runs_dir"])
extra = c2.text_area("โฟลเดอร์ผลประเมินเดิม (บรรทัดละหนึ่ง, ใช้ในหน้า Compare)",
                     "\n".join(raw.get("extra_eval_results_dirs") or []), height=80)
raw["extra_eval_results_dirs"] = [x.strip() for x in extra.splitlines() if x.strip()]
st.caption(f"path แบบ relative นับจากโฟลเดอร์ panel · ตอนนี้: data-pipeline = `{settings.data_pipeline_dir}`, "
           f"evaluation = `{settings.evaluation_dir}`")

st.subheader("Python ของแต่ละขั้น")
st.caption("แต่ละ repo ใช้ venv/conda คนละตัวได้ ใส่ path ของ python ในตัวนั้น เช่น `~/mlx-env/bin/python`")
cols = st.columns(4)
for col, stage in zip(cols, ("data", "perplexity", "train", "eval")):
    raw["python"][stage] = col.text_input(stage, raw["python"].get(stage) or sys.executable, key=f"py:{stage}")
raw["ludwig_command"] = st.text_input("คำสั่ง Ludwig", raw.get("ludwig_command", "ludwig"),
                                      help="ค่า 'ludwig' = เรียกผ่าน `<python ของ train> -m ludwig.cli`")

if st.button("🔎 ตรวจ Python ทั้งหมด"):
    for stage, py in raw["python"].items():
        mods = {"data": "transformers, bs4, pythainlp", "perplexity": "torch, transformers",
                "train": "ludwig", "eval": "torch, transformers, datasets"}[stage]
        code = f"import {mods}; import sys; print(sys.version.split()[0])"
        try:
            r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                st.success(f"{stage}: Python {r.stdout.strip()} · import {mods} ได้")
            else:
                st.error(f"{stage}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'error'}")
        except Exception as e:
            st.error(f"{stage}: {e}")

st.subheader("Environment variables")
st.caption("ทุกขั้นได้ค่าชุด `env` · `stage_env` เพิ่มเฉพาะขั้น (ค่าว่าง = ไม่ส่ง)")
env_yaml = st.text_area("YAML", yaml.safe_dump({"env": raw.get("env", {}), "stage_env": raw.get("stage_env", {})},
                                                allow_unicode=True, sort_keys=False), height=220)

if st.button("💾 บันทึก Settings", type="primary"):
    try:
        parsed = yaml.safe_load(env_yaml) or {}
        raw["env"] = parsed.get("env", {}) or {}
        raw["stage_env"] = parsed.get("stage_env", {}) or {}
        settings.save()
        st.success(f"บันทึกแล้วที่ {SETTINGS_FILE}")
    except yaml.YAMLError as e:
        st.error(f"YAML ไม่ถูกต้อง: {e}")
