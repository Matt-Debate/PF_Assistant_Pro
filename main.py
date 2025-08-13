# main.py
import streamlit as st
from auth import init_db, authenticate

init_db()

# Set session defaults
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None

ROLES = [None, "user", "admin"]

# --- Login function using real authentication ---
def login():
    st.header("Debate Helper Pro")
    st.subheader("Your personal Debate AI assistant")
    with st.form("login_form"):
        username = st.text_input("Username", key="input_username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            success, role = authenticate(username, password)
            if success:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = role
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Invalid credentials")

# --- Logout function ---
def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.rerun()

# --- Get current role ---
role = st.session_state.get("role", None)

# --- Define Pages ---
# --- User Pages ---
propmt_help = st.Page("pages_user/1_prompt_help.py", title="Prompt Help", icon="💡", default=(role == "user"))
get_research = st.Page("pages_user/2_get_research.py", title="Get Research", icon="🔍")
get_writing = st.Page("pages_user/3_get_writing.py", title="Get Writing", icon="✍️")
get_feedback = st.Page("pages_user/4_get_feedback.py", title="Get Feedback", icon="📝")

# --- Admin Pages ---
manage_users = st.Page("pages_admin/manage_users.py", title="Manage Users", icon="👥", default=(role == "admin"))
add_users = st.Page("pages_admin/add_users.py", title="Add Users", icon="➕")
io_history = st.Page("pages_admin/io_history.py", title="I/O History", icon="📄")
usage_data = st.Page("pages_admin/usage_data.py", title="Usage Data", icon="📊")
user_feedback = st.Page("pages_admin/user_feedback.py", title="User Feedback", icon="💬")
dev_testing = st.Page("pages_admin/5_tests.py", title="Dev Testing")

# --- Account Pages ---
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")
settings_page = st.Page("pages_account/settings.py", title="Settings", icon=":material/settings:")

account_pages = [logout_page, settings_page]

# --- Organize pages based on role ---
page_dict = {}

if not st.session_state.authenticated:
    # Only login page if not authenticated
    pg = st.navigation([st.Page("log_in.py")])
else:
    # Authenticated pages
    if role == "user":
        page_dict["Use the App"] = [get_writing, get_research, get_feedback]
    if role == "admin":
        page_dict["Admin"] = [manage_users, add_users, io_history, usage_data, user_feedback, dev_testing]
        page_dict["User Services"] = [get_writing, get_research, get_feedback]
    pg = st.navigation({"Account": account_pages} | page_dict)

# Run the navigation
pg.run()