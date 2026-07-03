"""
O Level Student Subject & Teacher Preference System
------------------------------------------------------
This script creates a SQLite database with tables for:
- Students
- Subjects
- Teachers
- Which teacher teaches which subject
- Which subject + teacher a student prefers

Run this file once to set up the database:
    python database.py
"""

import sqlite3

DB_NAME = "school.db"


def create_tables():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Enable foreign key constraints (off by default in SQLite)
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. Students
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            roll_number TEXT UNIQUE NOT NULL,
            grade_level TEXT,          -- e.g. "O Level Year 1"
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 2. Subjects
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT UNIQUE NOT NULL,   -- e.g. "Physics"
            subject_code TEXT UNIQUE             -- e.g. "PHY-5054"
        );
    """)

    # 3. Teachers
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            teacher_id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT,
            max_students INTEGER DEFAULT 30      -- optional capacity limit
        );
    """)

    # 4. Which teacher can teach which subject (many-to-many)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teacher_subjects (
            teacher_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            PRIMARY KEY (teacher_id, subject_id),
            FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id) ON DELETE CASCADE,
            FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
        );
    """)

    # 5. Student preferences: subject + preferred teacher + rank
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_preferences (
            preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            preferred_teacher_id INTEGER,        -- can be NULL if student has no preference
            priority INTEGER DEFAULT 1,          -- 1 = first choice, 2 = second choice, etc.
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
            FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE,
            FOREIGN KEY (preferred_teacher_id) REFERENCES teachers(teacher_id) ON DELETE SET NULL,
            UNIQUE (student_id, subject_id, priority)
        );
    """)

    conn.commit()
    conn.close()
    print(f"Database '{DB_NAME}' created successfully with all tables.")


def insert_sample_data():
    """Optional: adds a few example rows so you can test the structure."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Sample subjects
    subjects = [("Physics", "PHY-5054"), ("Mathematics", "MAT-4024"), ("Chemistry", "CHE-5070")]
    cursor.executemany("INSERT OR IGNORE INTO subjects (subject_name, subject_code) VALUES (?, ?);", subjects)

    # Sample teachers
    teachers = [("Ali Raza", "ali.raza@school.edu"), ("Sana Khan", "sana.khan@school.edu")]
    cursor.executemany("INSERT OR IGNORE INTO teachers (full_name, email) VALUES (?, ?);", teachers)

    # Sample student
    cursor.execute(
        "INSERT OR IGNORE INTO students (full_name, roll_number, grade_level, email) VALUES (?, ?, ?, ?);",
        ("Ahmed Hassan", "OL-2026-001", "O Level Year 2", "ahmed@example.com")
    )

    conn.commit()
    conn.close()
    print("Sample data inserted.")


if __name__ == "__main__":
    create_tables()
    insert_sample_data()
