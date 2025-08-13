import streamlit as st
from auth import add_user

st.header("👑 Admin Panel")
st.subheader("Add User")

with st.form("add_user_form"):
    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    new_role = st.selectbox("Role", ["user", "admin"])
    new_limit = st.number_input("Daily Limit", min_value=1, value=20)
    submitted = st.form_submit_button("Add User")
    if submitted:
        try:
            add_user(new_username, new_password, new_role, new_limit)
            st.success(f"User {new_username} added.")
        except Exception as e:
            st.error(f"Failed to add user: {e}")