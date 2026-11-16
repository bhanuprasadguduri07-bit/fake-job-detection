"""
Database module for the Fake Job Recruitment Detection System.
Provides SQLite connection management, schema initialization, and helper functions
with parameterized queries for security.
"""

import os
import sqlite3
from datetime import datetime
from config import Config

SQLITE_BUSY_TIMEOUT_MS = 30_000


def get_db_connection():
    """Establish a connection to the SQLite database with row factory enabled."""
    db_path = Config.DATABASE_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(
        db_path,
        timeout=SQLITE_BUSY_TIMEOUT_MS / 1000,
        check_same_thread=True
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db():
    """Create all required tables with constraints and indexes."""
    conn = get_db_connection()
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        cursor = conn.cursor()

        # 1. users
        cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

        # 2. user_requirements
        cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_requirements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE NOT NULL,
        preferred_title TEXT,
        preferred_company TEXT,
        required_skills TEXT,
        preferred_location TEXT,
        minimum_salary REAL DEFAULT 0,
        experience_level TEXT,
        job_type TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)

        # 3. job_submissions
        cursor.execute("""
    CREATE TABLE IF NOT EXISTS job_submissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        company_name TEXT NOT NULL,
        job_title TEXT NOT NULL,
        job_description TEXT NOT NULL,
        salary TEXT,
        job_location TEXT,
        required_skills TEXT,
        experience_requirements TEXT,
        recruiter_email TEXT,
        application_url TEXT,
        official_careers_url TEXT,
        submission_type TEXT DEFAULT 'text',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
    );
    """)

        # 4. uploaded_posters
        cursor.execute("""
    CREATE TABLE IF NOT EXISTS uploaded_posters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        submission_id INTEGER NOT NULL,
        file_path TEXT NOT NULL,
        file_name TEXT NOT NULL,
        file_size INTEGER,
        extracted_raw_text TEXT,
        ocr_engine_used TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (submission_id) REFERENCES job_submissions(id) ON DELETE CASCADE
    );
    """)

        # 5. ml_predictions
        cursor.execute("""
    CREATE TABLE IF NOT EXISTS ml_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        submission_id INTEGER UNIQUE NOT NULL,
        prediction_label TEXT NOT NULL,
        is_fake INTEGER NOT NULL,
        confidence_score REAL NOT NULL,
        fake_probability REAL NOT NULL,
        suspicious_indicators TEXT,
        explanation TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (submission_id) REFERENCES job_submissions(id) ON DELETE CASCADE
    );
    """)

        # 6. company_verification_results
        cursor.execute("""
    CREATE TABLE IF NOT EXISTS company_verification_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        submission_id INTEGER UNIQUE NOT NULL,
        company_name TEXT NOT NULL,
        verification_status TEXT NOT NULL,
        official_website_url TEXT,
        careers_page_url TEXT,
        vacancy_matched INTEGER DEFAULT 0,
        matching_vacancy_details TEXT,
        evidence TEXT,
        source_urls TEXT,
        explanation TEXT,
        last_verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (submission_id) REFERENCES job_submissions(id) ON DELETE CASCADE
    );
    """)

        # 7. monitored_websites
        cursor.execute("""
    CREATE TABLE IF NOT EXISTS monitored_websites (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        company_name TEXT NOT NULL,
        careers_url TEXT NOT NULL,
        status TEXT DEFAULT 'active',
        last_checked_at TIMESTAMP,
        last_status_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, company_name, careers_url),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)

        # 8. extracted_job_postings
        cursor.execute("""
    CREATE TABLE IF NOT EXISTS extracted_job_postings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        monitored_website_id INTEGER NOT NULL,
        job_title TEXT NOT NULL,
        job_location TEXT,
        department TEXT,
        salary TEXT,
        description TEXT,
        job_url TEXT,
        external_job_id TEXT,
        status TEXT DEFAULT 'open',
        first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(monitored_website_id, job_title, job_url),
        FOREIGN KEY (monitored_website_id) REFERENCES monitored_websites(id) ON DELETE CASCADE
    );
    """)

        # 9. job_match_results
        cursor.execute("""
    CREATE TABLE IF NOT EXISTS job_match_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        submission_id INTEGER,
        posting_id INTEGER,
        match_score REAL NOT NULL,
        title_match_score REAL,
        skills_match_score REAL,
        location_match_score REAL,
        experience_match_score REAL,
        salary_match_score REAL,
        match_breakdown TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (submission_id) REFERENCES job_submissions(id) ON DELETE CASCADE,
        FOREIGN KEY (posting_id) REFERENCES extracted_job_postings(id) ON DELETE CASCADE
    );
    """)

        # 10. monitoring_history
        cursor.execute("""
    CREATE TABLE IF NOT EXISTS monitoring_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        monitored_website_id INTEGER NOT NULL,
        check_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        vacancies_found INTEGER DEFAULT 0,
        new_vacancies_found INTEGER DEFAULT 0,
        reopened_vacancies_found INTEGER DEFAULT 0,
        status TEXT NOT NULL,
        notes TEXT,
        FOREIGN KEY (monitored_website_id) REFERENCES monitored_websites(id) ON DELETE CASCADE
    );
    """)

        # 11. notifications
        cursor.execute("""
    CREATE TABLE IF NOT EXISTS notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        company_name TEXT NOT NULL,
        job_title TEXT NOT NULL,
        match_percentage REAL,
        verification_status TEXT,
        source_link TEXT,
        message TEXT NOT NULL,
        is_read INTEGER DEFAULT 0,
        email_sent INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)

        # Add fields introduced after the initial schema without disturbing existing data.
        schema_updates = {
            "job_submissions": {
                "education": "TEXT",
                "contact_number": "TEXT",
                "official_website_url": "TEXT",
            },
            "company_verification_results": {
                "official_verification_status": "TEXT",
            },
        }
        for table_name, columns in schema_updates.items():
            existing_columns = {
                row["name"] for row in cursor.execute(f"PRAGMA table_info({table_name})")
            }
            for column_name, column_type in columns.items():
                if column_name not in existing_columns:
                    cursor.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
                    )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    print("Database schema successfully initialized.")


if __name__ == "__main__":
    init_db()
