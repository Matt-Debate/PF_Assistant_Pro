import streamlit as st
from auth import authenticate  # your auth function

st.set_page_config(page_title="Login - PF Hero", layout="centered")

st.title("Debate Helper Pro")
st.subheader("Your personal Debate AI assistant")

with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Log In")

    if submitted:
        success, role = authenticate(username, password)
        if success:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.role = role
            st.success(f"Welcome, {username}!")
            st.rerun()  # rerun app to load multipage after login
        else:
            st.error("Invalid credentials")

# Optional: clear session state when page loads (if you want fresh start each time)
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False