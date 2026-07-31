import os
import sys
import time
import requests
from sqlalchemy import text

# Add workspace root to python path to resolve imports
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from config.settings import settings
from database.connection import init_db, SessionLocal, engine
from services.embedding.sentence_transformer import SentenceTransformersEmbedding
from services.vector_store.postgres import PostgresVectorStore
from services.llm.ollama import OllamaLLMService
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.rag_pipeline import RAGPipeline


def check_services() -> bool:
    """
    Check if required infrastructure services (PostgreSQL and Ollama) are reachable.
    """
    # 1. Check PostgreSQL connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        print("[SUCCESS] PostgreSQL database is reachable.")
    except Exception as e:
        print(f"[ERROR] PostgreSQL is not reachable at: {settings.DATABASE_URL}")
        print(f"Details: {e}")
        print("\nPlease ensure your PostgreSQL service is running and pgvector is installed.\n")
        return False

    # 2. Check Ollama connection
    ollama_url = f"{settings.OLLAMA_BASE_URL}/api/tags"
    try:
        response = requests.get(ollama_url, timeout=3.0)
        if response.status_code == 200:
            print("[SUCCESS] Ollama service is reachable.")
            # Check if our target model is pulled
            models = [m.get("name") for m in response.json().get("models", [])]
            target_model = settings.LLM_MODEL
            # Ollama tags can include ':latest' or ':mini'
            model_installed = any(target_model in m for m in models)
            if model_installed:
                print(f"[SUCCESS] Ollama model '{target_model}' is installed.")
            else:
                print(f"[WARNING] Ollama model '{target_model}' was not found in installed tags: {models}")
                print(f"Please run: 'ollama pull {target_model}' in your terminal.")
                return False
        else:
            print(f"[ERROR] Ollama ping returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Ollama is not reachable at: {settings.OLLAMA_BASE_URL}")
        print(f"Details: {e}")
        print("\nPlease ensure Ollama is running ('ollama serve') and accessible.\n")
        return False

    return True


def run_e2e_test():
    print("=" * 60)
    print("STARTING END-TO-END RAG CLI INTEGRATION TEST")
    print("=" * 60)

    # Initialize Database Tables
    print("\nInitializing database schemas...")
    init_db()

    # 1. Initialize concrete services
    print("Loading SentenceTransformersEmbedding model (all-MiniLM-L6-v2)...")
    embed_service = SentenceTransformersEmbedding()

    # Create session
    session = SessionLocal()
    vector_store = PostgresVectorStore(db_session=session)

    print(f"Connecting to Ollama LLM Service (model: {settings.LLM_MODEL})...")
    llm_service = OllamaLLMService(timeout_seconds=180.0)

    # 2. Setup pipelines
    ingestion_pipe = IngestionPipeline(embedding_service=embed_service, vector_store=vector_store)
    rag_pipe = RAGPipeline(embedding_service=embed_service, vector_store=vector_store, llm_service=llm_service)

    # 3. Create mock OpenAPI document payload to ingest
    mock_spec_name = "cli_test_api_spec.json"
    mock_spec_json = """{
        "openapi": "3.0.0",
        "info": {
            "title": "Technical Support Test API",
            "version": "1.0.0",
            "description": "Standard API definition to verify pipeline grounding constraints."
        },
        "paths": {
            "/checkout": {
                "post": {
                    "summary": "Process customer checkout cart",
                    "description": "Submits items for purchase. Requires active token authorization.",
                    "parameters": [
                        {
                            "name": "X-Transaction-Key",
                            "in": "header",
                            "required": true,
                            "schema": {"type": "string"},
                            "description": "Unique transaction tracking key"
                        }
                    ],
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["cart_id", "payment_method"],
                                    "properties": {
                                        "cart_id": {"type": "string", "description": "UUID identifier of the shopping cart"},
                                        "payment_method": {"type": "string", "enum": ["credit_card", "paypal"]}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Checkout success response payload"
                        },
                        "401": {
                            "description": "Unauthorized response when X-Transaction-Key is missing"
                        }
                    }
                }
            }
        }
    }"""

    doc_id = None
    try:
        # Ingest the mock document
        print(f"\nIngesting mock API specification: '{mock_spec_name}'...")
        stats = ingestion_pipe.ingest_file(mock_spec_json.encode("utf-8"), mock_spec_name)
        doc_id = stats.get("document_id")
        print(f"Ingestion successful! Statistics: {stats}")

        # 4. Execute Q&A scenarios
        scenarios = [
            {
                "description": "Grounding Q&A Test (Content is in document)",
                "question": "What parameters and headers are required for the /checkout endpoint?"
            },
            {
                "description": "Hallucination Protection Test (Content is NOT in document)",
                "question": "What is the email address of the company CEO?"
            }
        ]

        for idx, sc in enumerate(scenarios):
            print("\n" + "=" * 50)
            print(f"SCENARIO {idx + 1}: {sc['description']}")
            print(f"Question: \"{sc['question']}\"")
            print("=" * 50)

            start_query = time.time()
            result = rag_pipe.query(sc["question"], top_k=3)
            elapsed = time.time() - start_query

            print(f"\nAnswer:\n{result['answer']}")
            
            print("\nSources Cited:")
            for s in result["sources"]:
                print(f"  - File: {s['filename']}, Similarity: {s['similarity_score']:.4f}, Endpoint: {s.get('http_method')} {s.get('api_path')}")
            
            print(f"\nMetrics: {result['metrics']}")
            print(f"Total loop time: {elapsed:.2f}s")

    finally:
        # 5. Cleanup database chunks to keep it clean
        if doc_id:
            print(f"\nCleaning up database, deleting document ID: '{doc_id}'...")
            vector_store.delete_document_chunks(doc_id)
            session.commit()
            print("Database cleanup completed successfully.")
            
        session.close()


if __name__ == "__main__":
    # Check if DB and Ollama are reachable before proceeding
    if check_services():
        run_e2e_test()
    else:
        print("\n[SKIP] E2E Integration test skipped due to missing service dependencies.")
        sys.exit(0)
