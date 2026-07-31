import logging
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from config.settings import settings

logger = logging.getLogger(__name__)

# SQLAlchemy 2.0 Declarative Base
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy database models."""
    pass


# Initialize SQLAlchemy Engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    pool_pre_ping=True,      # Automatically verify connection health before checking out
    pool_size=10,            # Sensible default pool size for connections
    max_overflow=20          # Max overflow connections when pool is full
)

# Initialize Session Factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def init_db() -> None:
    """
    Initialize the database, creating the vector extension if missing,
    and creating all tables defined in the SQLAlchemy metadata.
    """
    try:
        # Crucial for vector support: Enable pgvector extension in PostgreSQL
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
        
        # Import models here to register them with Base.metadata before creation
        import database.models  # noqa: F401
        
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully and pgvector extension is active.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise e


def get_db() -> Generator:
    """
    Dependency generator yielding database sessions.
    Cleans up and closes the session after request processing completes.
    
    Yields:
        SQLAlchemy Session object.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
