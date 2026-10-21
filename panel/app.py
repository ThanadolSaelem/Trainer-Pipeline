"""LLM Control Panel entry point:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="LLM Control Panel", page_icon="🧪", layout="wide")

pages = {
    "ภาพรวม": [
        st.Page("pages/home.py", title="Overview", icon="🏠", default=True),
    ],
    "ขั้นตอน": [
        st.Page("pages/data.py", title="1 · Data Cleaning", icon="🧹"),
        st.Page("pages/perplexity.py", title="2 · Perplexity", icon="📏"),
        st.Page("pages/train.py", title="3 · Train (Ludwig)", icon="🏋️"),
        st.Page("pages/evaluate.py", title="4 · Evaluate", icon="🎯"),
    ],
    "วิเคราะห์": [
        st.Page("pages/compare.py", title="Compare models", icon="📊"),
        st.Page("pages/jobs.py", title="All jobs", icon="🗂️"),
        st.Page("pages/settings.py", title="Settings", icon="⚙️"),
    ],
}
st.navigation(pages).run()
