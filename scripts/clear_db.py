import os
import sys

# Add parent workspace directory to Python system path to resolve relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.connection import SessionLocal
from sqlalchemy import text

def clear_db():
    print("Clearing all records from documents, document_chunks, and chat_history tables...")
    session = SessionLocal()
    try:
        # Cascade will delete all rows from document_chunks as well due to the foreign key constraint
        session.execute(text("TRUNCATE TABLE documents CASCADE;"))
        session.execute(text("TRUNCATE TABLE chat_history CASCADE;"))
        session.commit()
        print("Database cleared successfully.")
    except Exception as e:
        session.rollback()
        print(f"Error clearing database: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    clear_db()
