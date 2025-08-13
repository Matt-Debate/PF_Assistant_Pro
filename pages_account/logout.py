import streamlit as st

def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.success("You have been logged out.")
    st.rerun()

# Run logout immediately when this page is loaded
logout()
