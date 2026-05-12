import sqlite3
from datetime import datetime, timedelta
import random
from pathlib import Path

DB_PATH = Path('d:/Fitness_RAG_application/fitness_app.db')
USER_ID = "default_user"

# Connect to DB
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Make sure the user exists first
cursor.execute("INSERT OR IGNORE INTO profiles (user_id, name, goal) VALUES (?, ?, ?)", (USER_ID, "Athlete", "bulk"))

# Clear old logs for clean slate
cursor.execute("DELETE FROM logs WHERE user_id = ?", (USER_ID,))

# Generate 8 weeks (appx 2 months) of weekly check-ins
start_date = datetime.now() - timedelta(days=60)

weight = 170.0 # Starting weight (lbs equivalent translated later maybe, but let's just use raw numbers. The app handles units. Let's use kg for raw db = 77kg)
weight_kg = 75.0

bench = 80.0
squat = 100.0
deadlift = 120.0

for week in range(8):
    log_date = start_date + timedelta(days=week*7)
    
    # Simulate realistic progression (Bulk)
    weight_kg += random.uniform(0.2, 0.5)
    bench += random.uniform(1.0, 2.5)
    squat += random.uniform(2.0, 4.0)
    deadlift += random.uniform(2.0, 5.0)
    
    cursor.execute("""
        INSERT INTO logs (user_id, timestamp, weight_kg, body_fat_pct, pr_bench, bench_reps, pr_squat, squat_reps, pr_deadlift, deadlift_reps, workout_split)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        USER_ID, 
        log_date.strftime("%Y-%m-%d %H:%M:%S"), 
        round(weight_kg, 1), 
        15.0 - (week * 0.1), # BF drops slightly or stays same
        round(bench, 1),
        5,
        round(squat, 1),
        5,
        round(deadlift, 1),
        3,
        "3-day split"
    ))

conn.commit()
conn.close()

print(f"✅ Successfully injected 8 weeks of realistic training logs for {USER_ID}!")
