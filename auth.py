# auth.py

import sqlite3
import bcrypt
import streamlit as st
from datetime import datetime, date

DB_PATH = "users.db"

# ----------- DATABASE INITIALIZATION -----------

def init_query_log():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS query_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    success BOOLEAN,
                    feedback TEXT,
                    FOREIGN KEY(username) REFERENCES users(username)
                )''')
    conn.commit()
    conn.close()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    max_queries_per_day INTEGER DEFAULT 20
                )''')
    conn.commit()
    conn.close()
    init_query_log()

# ----------- AUTH & USER MGMT -----------

def add_user(username, password, role='user', max_queries_per_day=20):
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password_hash, role, max_queries_per_day) VALUES (?, ?, ?, ?)",
              (username, password_hash, role, max_queries_per_day))
    conn.commit()
    conn.close()

def authenticate(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()

    if result:
        stored_hash, role = result
        if bcrypt.checkpw(password.encode(), stored_hash):
            return True, role
    return False, None

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, role, CAST(max_queries_per_day AS INTEGER) FROM users")
    users = c.fetchall()
    conn.close()
    return users

def update_user_limit(username, new_limit):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET max_queries_per_day = ? WHERE username = ?", (new_limit, username))
    conn.commit()
    conn.close()

def delete_user(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()

# ----------- QUERY LOGGING -----------

def log_query(username, tool, success=True, feedback=None):
    timestamp = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO query_log (username, tool, timestamp, success, feedback) VALUES (?, ?, ?, ?, ?)",
              (username, tool, timestamp, int(success), feedback))
    conn.commit()
    conn.close()

def get_user_query_count_today(username):
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT COUNT(*) FROM query_log 
        WHERE username = ? AND date(timestamp) = ?
    ''', (username, today))
    count = c.fetchone()[0]
    conn.close()
    return count

def get_remaining_queries(username: str, tool: str = None, max_default: int = 20) -> int:
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Count tool usage
    if tool:
        c.execute('''
            SELECT COUNT(*) FROM query_log
            WHERE username = ? AND tool = ? AND date(timestamp) = ?
        ''', (username, tool, today))
    else:
        c.execute('''
            SELECT COUNT(*) FROM query_log
            WHERE username = ? AND date(timestamp) = ?
        ''', (username, today))

    count = c.fetchone()[0]

    # Get user limit
    c.execute('SELECT max_queries_per_day FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()

    limit = result[0] if result else max_default
    return max(0, limit - count)

def get_username():
    return st.session_state.get("username", "Unknown")

DB_PATH = "users.db"  # or whatever path you're using

def get_usage_logs(start_date=None, end_date=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    query = "SELECT username, tool, timestamp, success, feedback FROM query_log"
    params = []

    if start_date and end_date:
        query += " WHERE timestamp BETWEEN ? AND ?"
        params = [start_date, end_date]

    c.execute(query, params)
    rows = c.fetchall()
    conn.close()

    return rows