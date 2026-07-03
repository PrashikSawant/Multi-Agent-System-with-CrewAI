"""
app.py
Day 25 - Multi-Agent System with CrewAI
"""

import streamlit as st
from dotenv import load_dotenv
from crew import run_crew

load_dotenv()

st.set_page_config(page_title="Day 25 - Multi-Agent Crew", page_icon="🤖")
st.title("🤖 Day 25 — Multi-Agent Research Crew")
st.caption("3 agents collaborate: Researcher → Analyst → Writer")

with st.expander("👥 Meet your crew"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔎 Researcher**")
        st.caption("Searches the web and gathers verified facts")
    with col2:
        st.markdown("**🧠 Analyst**")
        st.caption("Extracts key insights from the research")
    with col3:
        st.markdown("**✍️ Writer**")
        st.caption("Turns insights into a clean report")

st.divider()

topic = st.text_input(
    "What should the crew research?",
    placeholder="e.g. AI agents in 2026, India startup ecosystem, quantum computing"
)

if st.button("🚀 Run Crew", type="primary"):
    if topic.strip() == "":
        st.warning("Please enter a research topic first.")
    else:
        st.info(
            "⏳ The crew is running — this takes 1-3 minutes since three agents "
            "are working sequentially. Check your terminal for live agent output.",
            icon="⏳"
        )
        with st.spinner(f"Researching '{topic}'..."):
            try:
                report = run_crew(topic)
                st.success("✅ Crew finished!")
                st.markdown("## 📄 Research Report")
                st.markdown(report)
            except Exception as e:
                st.error(f"Crew run failed: {e}")
                st.info(
                    "Common causes: missing SERPER_API_KEY in .env, "
                    "Groq rate limit hit, or CrewAI version mismatch."
                )