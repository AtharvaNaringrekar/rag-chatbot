import logging
import sys
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config.settings import settings
from database.connection import init_db
from api.router import router
from core.exceptions import TechnicalSupportBotException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("technical_support_bot")


# Create FastAPI App Instance
app = FastAPI(
    title=settings.APP_NAME,
    description="REST API backend for Technical Support Chatbot using pgvector RAG and EasyOCR Vision AI.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS Middleware
# Allows Streamlit frontend and local API clients to communicate safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to match client domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def on_startup():
    """
    Startup hook.
    Initializes PostgreSQL connection, creates vector extension, and generates schemas.
    """
    logger.info("Application starting up...")
    try:
        init_db()
        logger.info("Database initialized and connection verified.")
    except Exception as e:
        logger.critical(f"Failed to initialize database on startup: {e}", exc_info=True)
        sys.exit(1)


# Global Exception Handler for Application-Specific Exceptions
@app.exception_handler(TechnicalSupportBotException)
def technical_support_bot_exception_handler(request: Request, exc: TechnicalSupportBotException):
    logger.error(f"Global Handler: Encounted application exception: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.message}
    )


# Generic Exception Handler
@app.exception_handler(Exception)
def generic_exception_handler(request: Request, exc: Exception):
    logger.critical(f"Global Handler: Unhandled system error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred."}
    )


@app.get("/")
def root_redirect():
    """
    Root endpoint redirecting users to the API docs.
    """
    return {
        "message": f"Welcome to the {settings.APP_NAME} API!",
        "documentation": "/docs",
        "status": "active"
    }


if __name__ == "__main__":
    # Start ASGI Server
    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}...")
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
