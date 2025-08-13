import streamlit as st
from datetime import datetime
from auth import get_username, log_query, get_remaining_queries
from io import BytesIO
from docx import Document

# Import tool functions
from tools import tool_get_research

# --- Constants ---
TOOL_NAME = "get_research"
MAX_DAILY_QUERIES = 10  # adjust to your limit

# --- Helper for docx output ---
def create_docx_output(content):
    if not content:
        return None
    doc = Document()
    doc.add_paragraph(content)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# --- Streamlit UI ---
st.title("📚 Evidence Machine")
st.subheader("Let’s Cut Some Cards!")

username = get_username()
remaining = get_remaining_queries(username) or MAX_DAILY_QUERIES

if remaining <= 0:
    st.warning("You have no uses remaining today.")
    st.stop()

# --- Inputs ---
with st.form("research_form"):
    claim = st.text_area("Claim to find evidence for (required)", height=120)
    submitted = st.form_submit_button("Submit")

st.markdown(f"**You have {remaining} uses of {TOOL_NAME} remaining today.**")
st.info("⚠️ Save your result before searching again. New results will replace old ones.")

# --- Output ---
if submitted:
    if not claim.strip():
        st.error("Please enter a claim.")
        st.stop()

    with st.spinner("Cutting your card(s)..."):
        try:
            prompt = tool_get_research.build_perplexity_prompt(claim)
            raw_result = tool_get_research.query_perplexity(prompt)
            st.session_state["evidence_result"] = raw_result
            log_query(username, TOOL_NAME, True)
            st.success("✅ Evidence generated!")
        except Exception as e:
            st.error(f"❌ Error fetching evidence: {e}")
            log_query(username, TOOL_NAME, False)
            st.stop()

# --- Display Evidence ---
if "evidence_result" in st.session_state:
    raw_result = st.session_state["evidence_result"]
    with st.expander("🔍 Evidence Output", expanded=True):
        st.text_area("Cards", raw_result, height=300)




