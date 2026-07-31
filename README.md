# AI Technical Support Assistant using RAG + Vision AI

An enterprise-grade, production-quality AI Chatbot behaving like a Technical Support Engineer. The assistant helps developers, testers, and clients troubleshoot technical issues and understand APIs by answering queries and analyzing error screenshots **strictly** using uploaded company documentation.

Built with **Clean Architecture** principles in pure Python (no LangChain wrapping overhead) to ensure modularity, high performance, and easily swappable component layers.

---

## 🛠️ Tech Stack
* **Backend Framework**: FastAPI (Asynchronous REST API)
* **Frontend Dashboard**: Streamlit (Conversational chat window and file uploads)
* **Database**: PostgreSQL with `pgvector` (Vector similarity search using Cosine Distance)
* **Embeddings Model**: `all-MiniLM-L6-v2` via HuggingFace `sentence-transformers` (Local CPU/GPU inference)
* **Large Language Model**: Ollama (`phi3:mini`)
* **OCR Engine**: `EasyOCR` (Transcribes error stack logs from screenshots)
* **Vision Layout Service**: Pluggable interface (stubbed classifier determining CLI, IDE, or client contexts)
* **Formatting/Parsers**: `pypdf`, `python-docx`, `PyYAML`, and standard library JSON/YAML sniffs.

---

## 📐 Project Architecture & Data Flows

### Architecture
The project is designed using **Onion/Clean Architecture** principles. High-level business logic is decoupled from frameworks:
* **Core layer (`core/`)**: Holds entities (`models.py`), custom exceptions (`exceptions.py`), and base contracts (`interfaces/`). It has zero external package dependencies.
* **Services layer (`services/`)**: Implements interfaces for LLMs, Embeddings, OCR, Vision, and Database operations.
* **Pipelines layer (`pipeline/`)**: Coordinates workflow steps (Ingestion Pipeline, RAG Pipeline).
* **Delivery layers (`api/`, `dashboard/`)**: Manages external boundaries (FastAPI ASGI web endpoints, Streamlit dashboard browser).

```
technical_support_bot/
│
├── config/              # App configurations via pydantic-settings
├── core/                # Core domain models and interface abstractions
│   └── interfaces/      # ILLMService, IEmbeddingService, IVectorStore, etc.
├── database/            # SQLAlchemy connection pooling and JSONB pgvector schemas
├── services/            # Swappable concrete implementations of core interfaces
│   ├── parser/          # Custom PDF, Docx, Text, OpenAPI, Postman parsers
│   ├── embedding/       # sentence-transformers all-MiniLM-L6-v2 wrapper
│   ├── llm/             # Ollama phi3:mini HTTP client
│   ├── vector_store/    # pgvector database adapter
│   ├── ocr/             # EasyOCR transcribing engine
│   └── vision/          # Visual UI layout classification stub
├── pipeline/            # Ingestion and Q&A RAG workflow orchestrators
├── ingestion/           # Smart recursive splits and text normalizers
├── api/                 # FastAPI routes (ingest, chat, health) and schemas
├── dashboard/           # Streamlit conversational web panel
├── utils/               # Shared logging and filesystem helpers
├── tests/               # Unit test suites and CLI integration test runners
├── data/                # Local physical storage for raw manual uploads
└── main.py              # FastAPI uvicorn ASGI entrypoint
```

---

## 🚀 Installation & Setup

### Prerequisites
1. **Python 3.10+** installed.
2. **PostgreSQL** installed locally or reachable remotely.
   * Make sure `pgvector` is installed on your Postgres server (e.g. on Windows, download `pgvector` binaries and copy them into your Postgres `lib` and `share` directories).
3. **Ollama** installed and active.
   * Pull the phi3 model:
     ```bash
     ollama pull phi3:mini
     ```

### 1. Clone the Project & Setup Virtual Environment
Navigate to the project root and create a virtual environment:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate virtual environment (Linux / macOS)
source .venv/bin/activate
```

### 2. Install Dependencies
Install all package dependencies locked in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Configure Settings (.env)
Create a `.env` file in the root workspace folder (`d:/Technical support bot/.env`):
```env
# FastAPI Settings
API_HOST=127.0.0.1
API_PORT=8000
DEBUG=True

# Database Settings (Update connection credentials)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/tech_support_db
DB_ECHO=False

# Ollama LLM Settings
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=phi3:mini
LLM_TEMPERATURE=0.0

# Embedding Model
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# OCR & Vision
OCR_USE_GPU=False
VISION_MODEL_NAME=llava:latest
```

---

## 📋 Running the Project

### 1. Ingest Documentation (Admin command)
Before starting the chatbot, populate your PostgreSQL vector database using the production-ready CLI ingestion utility:

```bash
# Option A: Ingest a single document or API specification
python scripts/ingest.py docs/type1.pdf

# Option B: Ingest multiple specific documents together
python scripts/ingest.py docs/type1.pdf docs/steps.pdf docs/examples.pdf docs/type1.postman_collection.json

# Option C: Ingest an entire directory of documents recursively
python scripts/ingest.py docs/

# Option D: Force overwrite of duplicate files (bypasses prompt)
python scripts/ingest.py docs/ --force
```

#### Expected Ingestion Output:
```text
Resolved 2 file(s) for ingestion.

------------------------------------
Indexing:
Type1.pdf
Parser: PDF
Chunks: 126
Vectors Generated: 126
Completed
------------------------------------

------------------------------------
Indexing:
type1.postman_collection.json
Parser: Postman
Endpoints Found: 18
Chunks Generated: 84
Completed
------------------------------------

========================================
INGESTION RUN SUMMARY
========================================
Total Files: 2
Successful:  2
Failed:      0
Total Chunks: 210
Elapsed Time: 3.42 sec
========================================
```

### 2. Start the FastAPI Backend Server
Ensure your PostgreSQL database service is running and Ollama is active. Run the entrypoint:
```bash
python main.py
```
This compiles the PostgreSQL schemas, creates the pgvector table mapping, and boots up the API listener at `http://127.0.0.1:8000`. You can inspect the Swagger interactive playground at:
👉 **http://127.0.0.1:8000/docs**

### 3. Start the Streamlit Dashboard UI
Open a secondary terminal, activate the virtual environment, and run:
```bash
streamlit run dashboard/app.py
```
This launches a browser panel at:
👉 **http://localhost:8501**

---

## 🧪 Running Tests

### 1. Run Unit Tests (Offline & Fast)
To verify all parsers, splitters, templates, and adapters in isolation (all API pings, PyTorch weight loads, and database queries are mocked):
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 2. Run Q&A Grounding CLI Integration Test
To run a complete CLI integration test (ingests a mock spec, executes Q&A checks on local Ollama, and purges the database afterwards):
```bash
python tests/cli_rag_test.py
```

### 3. Run Multimodal OCR + Vision CLI Integration Test
To run E2E image diagnostic troubleshooting checks (draws mock screenshots, transcribes console lines, and queries pgvector):
```bash
python tests/cli_multimodal_test.py
```
