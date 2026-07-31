import os
import sys
import time
import argparse
from typing import List, Optional, Tuple

# Add parent workspace directory to Python system path to resolve relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from database.connection import init_db, SessionLocal
from database.models import DBDocument
from services.embedding.sentence_transformer import SentenceTransformersEmbedding
from services.vector_store.postgres import PostgresVectorStore
from pipeline.ingestion_pipeline import IngestionPipeline
from core.exceptions import TechnicalSupportBotException


def parse_args():
    parser = argparse.ArgumentParser(
        description="Production-grade command-line ingestion utility to parse and index documentation files (PDF, DOCX, TXT, OpenAPI, Postman) into pgvector."
    )
    parser.add_argument(
        "targets",
        nargs="+",
        help="Paths to one or more files, list of files, or directories to scan and ingest recursively."
    )
    parser.add_argument(
        "--on-duplicate",
        choices=["skip", "replace", "prompt"],
        default="prompt",
        help="Action to perform when a document with the same filename has already been indexed."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Shorthand flag to force re-indexing and replacing duplicate documents without prompting (equivalent to --on-duplicate replace)."
    )
    return parser.parse_args()


def scan_targets(targets: List[str]) -> List[str]:
    """
    Scans positional target arguments, walking directories recursively
    and compiling a list of all matching files.
    """
    supported_extensions = (".pdf", ".docx", ".txt", ".json", ".yaml", ".yml")
    all_files = []
    
    for target in targets:
        if os.path.isdir(target):
            # Recursively walk directory
            for root, _, files in os.walk(target):
                for file in files:
                    _, ext = os.path.splitext(file.lower())
                    if ext in supported_extensions:
                        all_files.append(os.path.join(root, file))
        elif os.path.isfile(target):
            _, ext = os.path.splitext(target.lower())
            if ext in supported_extensions:
                all_files.append(target)
            else:
                print(f"[WARNING] Skipping target '{target}': Unsupported file format.", file=sys.stderr)
        else:
            print(f"[WARNING] Skipping target '{target}': Path not found.", file=sys.stderr)
            
    # Remove duplicate paths
    return sorted(list(set(all_files)))


def handle_duplicate(db_session, vector_store: PostgresVectorStore, filename: str, action: str) -> str:
    """
    Check if a document exists and apply duplicate policy.
    Returns: 'skip', 'replace', or 'proceed'
    """
    existing = db_session.query(DBDocument).filter(DBDocument.filename == filename).first()
    if not existing:
        return "proceed"

    if action == "skip":
        return "skip"
    elif action == "replace":
        print(f"  - Duplicate detected: Deleting existing database vectors for '{filename}'...")
        vector_store.delete_document_chunks(existing.id)
        db_session.commit()
        return "proceed"
    
    # Prompt option
    if not sys.stdin.isatty():
        print(f"  - Non-interactive shell detected. Defaulting to skip for duplicate: '{filename}'.")
        return "skip"

    print(f"\n[WARNING] Document '{filename}' is already indexed in the database.")
    while True:
        choice = input("  Choose action: [1] Skip indexing, [2] Re-index and replace: ").strip()
        if choice == "1":
            return "skip"
        elif choice == "2":
            print(f"  - Deleting existing database vectors for '{filename}'...")
            vector_store.delete_document_chunks(existing.id)
            db_session.commit()
            return "proceed"
        else:
            print("  Invalid choice. Enter '1' or '2'.")


def ingest_file_safe(
    pipeline: IngestionPipeline, 
    db_session, 
    vector_store: PostgresVectorStore, 
    file_path: str,
    dup_action: str
) -> Tuple[bool, int]:
    """
    Safely reads, checks duplicates, validates headers, and indexes a single document.
    Returns: (is_success, chunk_count)
    """
    filename = os.path.basename(file_path)
    print("\n------------------------------------")
    print(f"Indexing:\n{filename}")

    # 1. Validation check: File existence & Size
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return False, 0
        
    try:
        # 2. Duplicate checking
        dup_decision = handle_duplicate(db_session, vector_store, filename, dup_action)
        if dup_decision == "skip":
            print("Status: Skipped (Already Indexed)")
            print("------------------------------------")
            return True, 0

        # 3. Read content
        with open(file_path, "rb") as f:
            content = f.read()

        # Validation check: Empty files
        if not content or len(content.strip()) == 0:
            print("Error: File is empty.")
            print("------------------------------------")
            return False, 0

        # 4. Run ingestion pipeline
        stats = pipeline.ingest_file(content, filename)
        
        # Format parser outputs for print summary
        file_type = stats.get("file_type", "unknown").upper()
        if file_type == "DOCX":
            parser_name = "Word (DOCX)"
        elif file_type == "PDF":
            parser_name = "PDF"
        elif file_type == "TXT":
            parser_name = "Text"
        elif file_type == "OPENAPI":
            parser_name = "OpenAPI"
        elif file_type == "POSTMAN":
            parser_name = "Postman"
        else:
            parser_name = file_type

        print(f"Parser: {parser_name}")
        
        # If postman, report endpoint count
        if file_type == "POSTMAN":
            # Chunks represent parsed requests
            print(f"Endpoints Found: {stats.get('total_chunks', 0)}")
            
        print(f"Chunks Generated: {stats.get('total_chunks', 0)}")
        print(f"Vectors Generated: {stats.get('total_chunks', 0)}")
        print("Completed")
        print("------------------------------------")
        
        # Commit database transaction
        db_session.commit()
        return True, stats.get("total_chunks", 0)

    except TechnicalSupportBotException as tbe:
        db_session.rollback()
        print(f"Error during parsing or ingestion: {tbe.message}")
        print("Status: Failed")
        print("------------------------------------")
        return False, 0
    except Exception as e:
        db_session.rollback()
        print(f"Unexpected error: {e}")
        print("Status: Failed")
        print("------------------------------------")
        return False, 0


def main():
    args = parse_args()
    
    # Force overrides duplicate action
    dup_action = "replace" if args.force else args.on_duplicate

    print("Initializing Database Schemas...")
    init_db()

    print("Loading SentenceTransformers Embedding Model (all-MiniLM-L6-v2)...")
    embed_service = SentenceTransformersEmbedding()
    
    db_session = SessionLocal()
    vector_store = PostgresVectorStore(db_session=db_session)
    ingestion_pipe = IngestionPipeline(embedding_service=embed_service, vector_store=vector_store)

    # Scan and resolve target files
    files_to_process = scan_targets(args.targets)
    total_files = len(files_to_process)
    
    if total_files == 0:
        print("\n[ERROR] No supported documentation files found to index. Exiting.")
        db_session.close()
        sys.exit(1)

    print(f"\nResolved {total_files} file(s) for ingestion.")
    
    start_run = time.time()
    successful = 0
    failed = 0
    total_chunks = 0

    for file_path in files_to_process:
        success, chunks = ingest_file_safe(
            ingestion_pipe, 
            db_session, 
            vector_store, 
            file_path, 
            dup_action
        )
        if success:
            successful += 1
            total_chunks += chunks
        else:
            failed += 1

    elapsed = time.time() - start_run
    db_session.close()

    print("\n" + "=" * 40)
    print("INGESTION RUN SUMMARY")
    print("=" * 40)
    print(f"Total Files: {total_files}")
    print(f"Successful:  {successful}")
    print(f"Failed:      {failed}")
    print(f"Total Chunks: {total_chunks}")
    print(f"Elapsed Time: {elapsed:.2f} sec")
    print("=" * 40)


if __name__ == "__main__":
    main()
