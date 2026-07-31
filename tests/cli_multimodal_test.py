import os
import sys
import time
import io
import requests
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import text

# Add workspace root to python path to resolve imports
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from config.settings import settings
from database.connection import init_db, SessionLocal
from services.embedding.sentence_transformer import SentenceTransformersEmbedding
from services.vector_store.postgres import PostgresVectorStore
from services.llm.ollama import OllamaLLMService
from services.ocr.easy_ocr import EasyOCRService
from services.vision.vision_stub import VisionLLMStubService
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.rag_pipeline import RAGPipeline


def create_dummy_image(text_lines: list, filename: str) -> bytes:
    """
    Utility to programmatically render a mock screenshot containing error lines.
    Saves the file to tests/data/ for inspection and returns the raw bytes.
    """
    os.makedirs("tests/data", exist_ok=True)
    filepath = os.path.join("tests/data", filename)
    
    # Create dark theme terminal-like background
    img = Image.new("RGB", (600, 200), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    
    # Draw simple text line by line (using default system font)
    y_offset = 20
    for line in text_lines:
        # Drawing in bright green/white code style colors
        draw.text((20, y_offset), line, fill=(0, 255, 0))
        y_offset += 25
        
    img.save(filepath, "PNG")
    
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def run_multimodal_test():
    print("=" * 70)
    print("STARTING END-TO-END MULTIMODAL OCR + VISION RAG TEST")
    print("=" * 70)

    # Initialize Database Tables
    print("\nInitializing database schemas...")
    init_db()

    # Load concrete services
    embed_service = SentenceTransformersEmbedding()
    session = SessionLocal()
    vector_store = PostgresVectorStore(db_session=session)
    llm_service = OllamaLLMService(timeout_seconds=180.0)
    
    # OCR & Vision services
    ocr_service = EasyOCRService()
    vision_service = VisionLLMStubService()

    # Pipelines
    ingestion_pipe = IngestionPipeline(embedding_service=embed_service, vector_store=vector_store)
    rag_pipe = RAGPipeline(
        embedding_service=embed_service,
        vector_store=vector_store,
        llm_service=llm_service,
        ocr_service=ocr_service,
        vision_service=vision_service
    )

    # Ingest a sample API troubleshooting document
    knowledge_base_txt = (
        "## Technical Support Troubleshooting Guidelines\n\n"
        "### Section A: Connection Refused (port 5432 / PostgreSQL)\n"
        "If you see 'connection refused' or connection failures on port 5432, "
        "PostgreSQL is either stopped or listening on the wrong interface.\n"
        "Resolution:\n"
        "1. Check pg status: 'sudo systemctl status postgresql'\n"
        "2. Ensure pg_hba.conf permits inbound TCP loops.\n\n"
        "### Section B: CORS Policy Blocked\n"
        "If browser logs show CORS policy blocks on API requests:\n"
        "Resolution:\n"
        "1. Configure FastAPI CORS middleware.\n"
        "2. Add allow_origins=['*'] or matching client domain in FastAPI main.py.\n\n"
        "### Section C: 401 Unauthorized Credentials\n"
        "Postman or CLI tests yielding HTTP 401 response code indicate "
        "an invalid or expired 'X-Transaction-Key' header.\n"
        "Resolution:\n"
        "1. Verify key values in the headers section.\n"
        "2. Check token expirations in database 'tokens' tables."
    )

    doc_id = None
    try:
        print("\nIngesting Troubleshooting Reference Manual...")
        stats = ingestion_pipe.ingest_file(knowledge_base_txt.encode("utf-8"), "troubleshooting_guide.txt")
        doc_id = stats.get("document_id")
        print(f"Ingestion successful! Document ID: {doc_id}")

        # Define 3 multimodal test scenarios
        scenarios = [
            {
                "name": "Terminal CLI Error Screenshot",
                "filename": "terminal_error.png",
                "image_text": [
                    "psql: error: connection to server on socket failed",
                    "FATAL: connection refused on port 5432"
                ],
                "prompt": "How to resolve this database connection error showing in terminal?"
            },
            {
                "name": "Postman 401 Unauthorized Screenshot",
                "filename": "postman_unauthorized.png",
                "image_text": [
                    "POST /api/v1/charge - HTTP 401",
                    "{\"error\": \"Unauthorized - transaction key invalid\"}"
                ],
                "prompt": "Postman requests are failing with this error, what is the fix?"
            },
            {
                "name": "Browser DevTools CORS Screenshot",
                "filename": "cors_error.png",
                "image_text": [
                    "Access to XMLHttpRequest at gateway blocked by CORS policy",
                    "No 'Access-Control-Allow-Origin' header is present"
                ],
                "prompt": "Users are reporting web client loading issues, check this devtools console screen."
            }
        ]

        for sc in scenarios:
            print("\n" + "=" * 60)
            print(f"RUNNING SCENARIO: {sc['name']}")
            print(f"User Prompt: \"{sc['prompt']}\"")
            print("=" * 60)

            # Generate the mock screenshot image
            img_bytes = create_dummy_image(sc["image_text"], sc["filename"])
            print(f"Mock image '{sc['filename']}' generated and saved in tests/data/.")

            # Process the image query
            start_time = time.time()
            result = rag_pipe.query_image(img_bytes, user_prompt=sc["prompt"], top_k=3)
            elapsed = time.time() - start_time

            print("\n--- Pipeline Extraction Details ---")
            print(f"Transcribed OCR Text:\n{result['extracted_text']}")
            print(f"Vision Classified Layout: {result['vision_description']}")
            
            print("\n--- Grounded Answer ---")
            print(result["answer"])

            print("\n--- Citations ---")
            for s in result["sources"]:
                print(f"  - File: {s['filename']}, Similarity: {s['similarity_score']:.4f}, Section: {s.get('section')}")
                
            print(f"\nTime taken: {elapsed:.2f}s | Metrics: {result['metrics']}")

    finally:
        if doc_id:
            print(f"\nCleaning up database, deleting document ID: '{doc_id}'...")
            vector_store.delete_document_chunks(doc_id)
            session.commit()
            print("Database cleanup completed.")
            
        session.close()


if __name__ == "__main__":
    # Check services availability (reusing the ping check concept)
    # We check if Postgres is running before executing
    try:
        session = SessionLocal()
        session.execute(text("SELECT 1;"))
        session.close()
        run_multimodal_test()
    except Exception as e:
        print(f"[SKIP] Integration tests skipped: Database connection failed: {e}")
