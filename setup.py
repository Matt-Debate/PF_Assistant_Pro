# setup_db.py

from auth import init_db, add_user

# Initialize DB and create tables
init_db()

# Add admin user
add_user("admin", "password123", role="admin", max_queries_per_day=1000)

print("Database initialized and admin user created.")
