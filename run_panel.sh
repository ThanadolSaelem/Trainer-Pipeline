#!/usr/bin/env bash
# Start the LLM Control Panel. First run creates .venv and installs the panel's (light) dependencies.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if [ ! -x .venv/bin/python ]; then
  echo "[panel] creating .venv with $($PY --version)"
  "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi

if [ ! -f panel.yaml ]; then
  echo "[panel] tip: set each stage's Python interpreter on the Settings page"
fi

exec .venv/bin/streamlit run app.py --server.headless false --browser.gatherUsageStats false "$@"
