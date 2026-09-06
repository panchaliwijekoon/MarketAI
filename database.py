import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATABASE_FILE = BASE_DIR / "marketai.db"


def get_connection():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.row_factory = sqlite3.Row

    # Enables SQLite foreign-key checks
    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


def column_exists(
    cursor,
    table_name,
    column_name
):

    columns = cursor.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    return any(
        column[1] == column_name
        for column in columns
    )


def create_database():

    connection = get_connection()

    cursor = connection.cursor()


    # ===================================
    # USERS
    # ===================================

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            full_name TEXT NOT NULL,

            email TEXT NOT NULL UNIQUE,

            password_hash TEXT NOT NULL,

            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP

        )
        """
    )


    # ===================================
    # BUSINESS PROFILES
    # ===================================

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS business_profiles (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER,

            company_name TEXT NOT NULL,

            industry TEXT NOT NULL,

            product_name TEXT NOT NULL,

            hs_code TEXT,

            export_capacity REAL,

            preferred_region TEXT,

            experience_level TEXT,

            contact_email TEXT,

            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (user_id)
                REFERENCES users(id)

        )
        """
    )


    # If the table already existed before
    # user_id was added, add it now.

    if not column_exists(
        cursor,
        "business_profiles",
        "user_id"
    ):

        cursor.execute(
            """
            ALTER TABLE business_profiles
            ADD COLUMN user_id INTEGER
            """
        )


    # ===================================
    # ANALYSIS REPORTS
    # ===================================

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_reports (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            profile_id INTEGER NOT NULL,

            product_name TEXT,

            hs_code TEXT,

            top_market TEXT,

            predicted_demand REAL,

            predicted_growth REAL,

            market_score REAL,

            rankings_json TEXT,

            weights_json TEXT,

            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (profile_id)
                REFERENCES business_profiles(id)

        )
        """
    )


    connection.commit()

    connection.close()