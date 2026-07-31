# Developer & Architecture Guide

Welcome to the AI Technical Support Assistant codebase. This guide details the software architecture, design patterns, and instructions for extending or customizing the RAG + Vision AI pipeline.

---

## 📐 Architecture Framework
The codebase is structured around **Clean Architecture (Onion Architecture)**. Business rules are placed in the core layer, and framework integrations (FastAPI, SQLAlchemy, EasyOCR) are pushed to the outer service boundaries.

```
+-----------------------------------------------------------+
| API / Streamlit Delivery (FastAPI, Streamlit UI)          |
|   +-------------------------------------------------------+
|   | Services / Infrastructure (pgvector, EasyOCR, Ollama) |
|   |   +---------------------------------------------------+
|   |   | Pipelines Orchestrators (Ingestion, RAG Loop)     |
|   |   |   +-----------------------------------------------+
|   |   |   | Domain Core (Models, Exceptions, Contracts)   |
|   |   |   +-----------------------------------------------+
|   |   +---------------------------------------------------+
|   +-------------------------------------------------------+
+-----------------------------------------------------------+
```

### Layer Responsibilities

1. **Core Layer (`core/`)**
   * **Domain Entities (`models.py`)**: Data Transfer Objects (DTOs) describing documents, vector chunks, OCR results, and similarity query results.
   * **Interfaces (`interfaces/`)**: Abstract Base Classes (ABCs) defining execution boundaries (e.g. `IOCRService`, `IVectorStore`).
   * **Exceptions (`exceptions.py`)**: Strict, application-scoped error envelopes (e.g., `UnsupportedFormatException`, `LLMException`).
   * *Dependency Rule*: This package has **zero external package dependencies** (only standard libraries).

2. **Ingestion & Processing (`ingestion/`)**
   * **Text Normalization (`normalizers.py`)**: Standardizes input strings, handles carriage returns, and collapses duplicate spacings.
   * **Smart Chunker (`chunkers.py`)**: Implements recursive paragraph-aware splits preserving document structure margins and overlaps.

3. **Services Layer (`services/`)**
   * Houses concrete implementations of the abstract interfaces.
   * **Parsers (`services/parser/`)**: Implements `openapi.py` (resolving `$ref` loops to keep schema contexts intact), `postman.py` (recursive directory walks), and `doc_parsers.py` (PDF and DOCX extraction).
   * **Model Adapters**: `embedding/` (sentence-transformers), `llm/` (Ollama Rest client), `ocr/` (EasyOCR), and `vision/` (visual UI classifier).

4. **Pipelines Layer (`pipeline/`)**
   * Orchestrates multi-step workflows. `IngestionPipeline` connects Sniffers -> Parsers -> Chunker -> Embedding -> Vector Store. `RAGPipeline` coordinates Prompt Templates -> Embedding -> Cosine distance search -> Ollama LLM execution.

5. **Delivery Layer (`api/`, `dashboard/`, `main.py`)**
   * FastAPI async routers (`router.py`) and Streamlit visual interface (`app.py`).

---

## 🎨 Core Design Decisions & SOLID Compliance

### 1. Creational Factory Sniffing (OpenAPI vs Postman)
To avoid manual configuration overhead, the `DocumentParserFactory` performs content sniffing on JSON uploads:
```python
# snippet from services/parser/factory.py
if "info" in data and "paths" in data:
    return OpenAPIParser()
elif "info" in data and "item" in data:
    return PostmanCollectionParser()
```
This isolates the JSON parser selection dynamically from the caller.

### 2. Dependency Injection via FastAPI
We utilize FastAPI's built-in dependency injection (`Depends`) to inject transient PostgreSQL session contexts into persistent singleton pipelines.
```python
def get_rag_pipeline(db: Session = Depends(get_db)) -> RAGPipeline:
    vector_store = PostgresVectorStore(db_session=db)
    return RAGPipeline(
        embedding_service=embedding_service,  # Singleton preloaded in memory
        vector_store=vector_store,
        llm_service=llm_service,
        ocr_service=ocr_service,              # Singleton
        vision_service=vision_service         # Singleton
    )
```

### 3. Performance Optimization (GPU/CPU Timing)
To prevent PyTorch model reload lag, the SentenceTransformer and EasyOCR weights are loaded once in global memory during server startup. This reduces subsequent chat and ingestion latency to sub-second processing (excluding Ollama inference latency).

---

## 🛠️ How to Extend the Codebase

### A. Swapping out PostgreSQL for a Vector Database (e.g. Qdrant)
1. Navigate to `services/vector_store/` and create `qdrant.py`.
2. Implement the `IVectorStore` interface:
   ```python
   from core.interfaces.vector_store import IVectorStore
   
   class QdrantVectorStore(IVectorStore):
       def save_document(self, document): ...
       def save_document_chunks(self, chunks): ...
       def similarity_search(self, query_vector, limit=5, filter_metadata=None): ...
       def delete_document_chunks(self, document_id): ...
   ```
3. Update `api/router.py` dependency helpers (`get_rag_pipeline`, `get_ingestion_pipeline`) to instantiate `QdrantVectorStore` instead of `PostgresVectorStore`. No pipelines or routes change!

### B. Binding a Real Vision LLM (e.g. Gemini / LLaVA REST)
1. Navigate to `services/vision/` and create a new service.
2. Implement `IVisionService` sending the image bytes to your external REST API.
3. Replace the `VisionLLMStubService` instance in `api/router.py` with your new class.
