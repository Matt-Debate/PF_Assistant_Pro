import streamlit as st
import pandas as pd
from auth import get_all_users, delete_user, update_user_limit

st.header("👥 Manage Users")

# --- Load user data from DB ---
users = get_all_users()  # (username, role, limit)
df = pd.DataFrame(users, columns=["Username", "Role", "Daily Limit"])

# --- Search ---
search = st.text_input("🔍 Search by username", "")
if search:
    df = df[df["Username"].str.contains(search, case=False)]

# --- Editable Table ---
st.subheader("User Table")
edited_df = st.data_editor(
    df,
    column_config={
        "Daily Limit": st.column_config.NumberColumn(min_value=1),
        "Role": st.column_config.SelectboxColumn(options=["user", "admin"]),
    },
    use_container_width=True,
    disabled=["Username"],
    key="user_editor"
)

# --- Save Changes ---
if st.button("💾 Save Changes"):
    changes = []
    for i in range(len(df)):
        original = df.iloc[i]
        edited = edited_df.iloc[i]
        if original["Daily Limit"] != edited["Daily Limit"] or original["Role"] != edited["Role"]:
            update_user_limit(edited["Username"], edited["Daily Limit"])
            # update_user_role(edited["Username"], edited["Role"])  # Uncomment if implemented
            changes.append(edited["Username"])

    if changes:
        st.success(f"Updated: {', '.join(changes)}")
        st.rerun()
    else:
        st.info("No changes detected.")

# --- Inline Delete with Confirmation ---
st.subheader("🗑️ Delete Users")

for i, row in df.iterrows():
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        st.write(f"**{row['Username']}** ({row['Role']})")
    with col2:
        confirm = st.checkbox(f"Confirm delete", key=f"confirm_{row['Username']}")
    with col3:
        if st.button("❌ Delete", key=f"delete_{row['Username']}", disabled=not confirm):
            delete_user(row["Username"])
            st.warning(f"Deleted user: {row['Username']}")
            st.rerun()