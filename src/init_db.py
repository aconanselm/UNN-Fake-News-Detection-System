"""
UNN Community Fake News Detection System
Database initialization (run ONCE)

Creates your SQLite database file from the schema you designed in
database/schema.sql.

Run from the project root with the virtual environment activated:
    python src/init_db.py
"""

import os
import sqlite3

DB_PATH = 'database/unn_fnd.db'
SCHEMA_PATH = 'database/schema.sql'

if os.path.exists(DB_PATH):
    print(f"Database already exists at {DB_PATH} - no changes made.")
    print("To start fresh, delete that file and rerun this script.")
else:
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    print(f"Database created at {DB_PATH}")
    print("Tables: users, admin, submissions, fake_news_detection, feedback")
