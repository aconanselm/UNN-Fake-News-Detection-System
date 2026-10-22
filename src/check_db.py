"""
UNN Community Fake News Detection System
Database check - is the database set up, and what is in it?

Run from the project root with the virtual environment activated:
    python src/check_db.py
"""

import os
import sqlite3

DB_PATH = 'database/unn_fnd.db'
EXPECTED_TABLES = ['users', 'admin', 'submissions', 'fake_news_detection', 'feedback']

if not os.path.exists(DB_PATH):
    print(f"NOT SET UP: {DB_PATH} does not exist.")
    print("Fix: run  python src/init_db.py")
    raise SystemExit

conn = sqlite3.connect(DB_PATH)
existing = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]

print(f"Database file found: {DB_PATH} ({os.path.getsize(DB_PATH) / 1024:.0f} KB)\n")

missing = [t for t in EXPECTED_TABLES if t not in existing]
print(f"{'Table':<24}{'Rows':>6}")
print("-" * 30)
for t in EXPECTED_TABLES:
    if t in existing:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"{t:<24}{n:>6}")
    else:
        print(f"{t:<24}{'MISSING':>6}")

if missing:
    print(f"\nPROBLEM: missing tables: {', '.join(missing)}")
    print("Fix: delete database/unn_fnd.db, then run  python src/init_db.py")
else:
    print("\nAll 5 tables exist. The database is set up correctly.")

recent = conn.execute(
    """SELECT submission_id, category, verdict, confidence_score, substr(submitted_text, 1, 45)
       FROM submissions ORDER BY submission_id DESC LIMIT 5""").fetchall()
if recent:
    print("\nLast 5 submissions:")
    for r in recent:
        print(f"  #{r[0]}  {r[1]:<10} {r[2]:<5} {r[3]}%   {r[4]}...")

conn.close()
