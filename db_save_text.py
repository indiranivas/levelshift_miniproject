import sqlite3
import pandas as pd

def create_db():
    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            file_path TEXT,
            content TEXT,
            name TEXT,
            email TEXT,
            job_description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

def save_to_db(file_name, file_path, content, name, email, job_description):
    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO files (file_name, file_path, content, name, email, job_description)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (file_name, file_path, content, name, email, job_description))

    conn.commit()
    conn.close()

def save_job(title, description):
    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO jobs (title, description)
        VALUES (?, ?)
    """, (title, description))

    conn.commit()
    conn.close()

def get_jobs():
    conn = sqlite3.connect("data.db")
    
    df = pd.read_sql_query("SELECT * FROM jobs", conn)
    
    conn.close()
    return df

def get_all_data():
    conn = sqlite3.connect("data.db")
    
    df = pd.read_sql_query("SELECT * FROM files", conn)
    
    conn.close()
    return df

def delete_db():
    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM files")

    conn.commit()
    conn.close()