import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATABASE_FILE = BASE_DIR / "marketai.db"


def get_connection():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    connection.row_factory = sqlite3.Row

    return connection


def create_database():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS business_profiles (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            company_name TEXT NOT NULL,

            industry TEXT NOT NULL,

            product_name TEXT NOT NULL,

            hs_code TEXT,

            export_capacity REAL,

            preferred_region TEXT,

            experience_level TEXT,

            contact_email TEXT,

            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP

        )
        """
    )


    connection.commit()

    connection.close()