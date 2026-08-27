# AI Technical Support Assistant (Spintly API RAG Chatbot)
### *Internship Project at Spintly*

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql)](https://www.postgresql.org)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A5%97%20Hugging%20Face-Transformers-FFD21E?style=for-the-badge)](https://huggingface.co)
[![Ollama](https://img.shields.io/badge/Ollama-Run%20Local-black?style=for-the-badge)](https://ollama.com)

---

## 1. Project Overview
This repository hosts the **AI Technical Support Assistant**, a Retrieval-Augmented Generation (RAG) chatbot and developer API assistant. The assistant is built strictly on the **Spintly API Documentation** as its knowledge base. It is designed to act like an experienced technical support engineer, helping developers, testers, and integration clients troubleshoot system errors, understand authentication procedures, and configure API payloads.

To guarantee zero-hallucination responses for critical API operations:
- **Embeddings**: Sentences are vectorized using Hugging Face's `all-MiniLM-L6-v2` transformer model (384 dimensions) to represent text semantically.
- **PostgreSQL + pgvector**: A local PostgreSQL database equipped with the `pgvector` extension serves as a high-performance vector store, calculating cosine distances to retrieve relevant documentation passages.
- **RAG Orchestrator**: The RAG pipeline classifies user query intents (such as generating oauth tokens or creating users), retrieves related spec chunks, isolates matching endpoints, and constructs a grounded prompt.
- **Ollama (`qwen2.5:1.5b`)**: A local Ollama daemon serves as the LLM reasoning compiler to compile the final response strictly within the bounds of retrieved documentation.
- **Streamlit Chat Dashboard**: A web interface providing a clean ChatGPT-like user experience, complete with screenshot/image uploading, real-time status check badges, and a "Developer Mode" to view vector distance citations and execution latencies.

---

## 2. Key Features
- **RAG-Based Grounded Q&A**: Generates precise instructions strictly bound to authorized documentation.
- **API Spec Retrieval & Mapping**: Decodes complex endpoints, methods, header parameters, and JSON payloads.
- **Query Classification & Context Isolation**: Identifies targeted endpoints (like user creation or permissions) and filters out irrelevant vector noise.
- **Response Validation & Fallback Guardrails**: Verifies if generated code blocks have correct syntax and contain all required properties. If a check fails, the pipeline triggers a safe, deterministic, Postman-compliant instruction step.
- **Source & Citation Info**: Lists exact document names, sections, pages, and cosine similarity scores for matching chunks (visible in Developer Mode).
- **Developer Mode Toggle**: Exposes real-time system latency metrics (Embedding, Retrieval, LLM generation) and reference citations.
- **Image/Vision Troubleshooting**: Supports uploading screenshot attachments, transcribing code errors via `EasyOCR`, and analyzing visual layout segments via a pluggable vision Stub layer.

---

## 3. System Architecture
The application flow executes through the following path:

```
User Query / Screenshot
  │
  ▼
Streamlit Dashboard (dashboard/app.py) ──[HTTP POST]──► FastAPI Backend (main.py)
                                                               │
                                                               ▼
                                                       APIRouter (api/router.py)
                                                               │
                                                               ▼
                                                    RAGPipeline (pipeline/rag_pipeline.py)
                                                               │
                                         ┌─────────────────────┴─────────────────────┐
                                         ▼                                           ▼
                                 Embedding Model                            PostgreSQL + pgvector
                          (SentenceTransformers Embedding)                   (Similarity Search)
                                         │                                           │
                                         └─────────────────────┬─────────────────────┘
                                                               ▼
                                                       Grounded Prompt
                                                               │
                                                               ▼
                                                       Ollama LLM Daemon
                                                       (qwen2.5:1.5b Model)
                                                               │
                                                               ▼
                                                       Response Validation
                                                               │
                                                               ▼
                                                         Final Answer
```

---

## 4. Project Structure
```
technical_support_bot/
├── api/                 # FastAPI routes, schemas, and request handlers
│   ├── router.py        # Preloads model services globally and registers routes
│   └── schemas.py       # Pydantic schemas validating API payloads
├── config/              # Strong-typed configurations loading from .env
│   └── settings.py      # Core Settings class mapping environment variables
├── core/                # Clean Architecture core interfaces and exceptions
│   ├── interfaces/      # Service blueprints (IEmbeddingService, ILLMService, etc.)
│   ├── exceptions.py    # Custom domain exceptions
│   └── models.py        # Agnostic domain models
├── dashboard/           # Chat interface web application panels
│   └── app.py           # Streamlit UI dashboard and composer view
├── database/            # Database connection pools and model schemas
│   ├── connection.py    # Creates engine pool and initializes extension tables
│   └── models.py        # DBDocument, DBDocumentChunk database schemas
├── data/                # Documentation directory
│   └── raw/             # Raw technical manuals scanned during ingestion
├── ingestion/           # Document splitters and text normalization
├── pipeline/            # RAG flow orchestrators and system prompt templates
│   ├── ingestion_pipeline.py # Formats files and saves embeddings to pgvector
│   ├── prompts.py       # System prompt templates and operation classifications
│   └── rag_pipeline.py  # Orchestrates end-to-end classification, search, and generation
├── services/            # Swappable concrete service implementations
│   ├── embedding/       # SentenceTransformers embedding adapter
│   ├── llm/             # Ollama client session manager
│   ├── ocr/             # EasyOCR text extractor
│   ├── parser/          # OpenAPI, Postman, PDF, Word, and TXT document parser classes
│   ├── vector_store/    # Postgres pgvector similarity search logic
│   └── vision/          # Vision stub classifier
├── scripts/             # Python scripting CLI tools
│   ├── clear_db.py      # Clears database documents and chunk vectors
│   └── ingest.py        # Ingestion script to scan and write raw documents
├── tests/               # Pytest unit and integration test suite
│   ├── cli_multimodal_test.py # Multimodal end-to-end integration test
│   ├── cli_rag_test.py  # RAG end-to-end integration test
│   └── test_*.py        # Unit tests verifying individual layers
├── main.py              # FastAPI application server entrypoint
└── RAG_Chatbot_Colab.ipynb # Google Colab T4 GPU setup and benchmark notebook
```

---

## 5. Technologies Used
- **Python (>=3.10)**: Core language framework.
- **FastAPI (>=0.110.0)**: High-performance asynchronous REST API backend.
- **Streamlit (>=1.32.0)**: Clean chatbot interface panel.
- **PostgreSQL (>=12)**: Relational database hosting.
- **pgvector (>=0.2.5)**: PostgreSQL vector similarity extension.
- **SQLAlchemy (>=2.0.0)**: ORM connection manager.
- **Sentence Transformers (>=2.5.0)**: Hugging Face vector embedding adapter.
- **Ollama**: Local inference daemon engine.
- **qwen2.5:1.5b**: Reasoning LLM model.
- **EasyOCR (>=1.7.1)**: Image text reader engine.
- **Pillow (>=10.2.0)**: Image pixel handler.

---

## 6. Requirements / Prerequisites

### Software Required
- **Git**: For cloning the repository.
- **Python (3.10 - 3.11)**: Target runtime.
- **PostgreSQL (v12 - v16)**: Database backend with pgvector installed.
- **Ollama**: Installed and running on the host system.

### Models Required
- **qwen2.5:1.5b**: Must be pulled in Ollama (`ollama pull qwen2.5:1.5b`).

### Hardware Requirements
- **NVIDIA GPU (Optional but Recommended)**: Accelerates local SentenceTransformers embedding and Ollama generation.
- **CPU Execution**: Supported by default. The system runs fully on CPU if CUDA is unavailable, though response generation times will be slower.
- **T4 GPU**: Required on Google Colab to complete the performance benchmark within the <=10-second target.

---

## 7. Installation on a Fresh Windows PC

Follow this exact sequence of commands to set up the environment:

### 1. Clone the Repository
```powershell
git clone https://github.com/AtharvaNaringrekar/rag-chatbot.git
cd rag-chatbot
```

### 2. Configure Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Package Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Install PostgreSQL & pgvector
- Install PostgreSQL (v14 is recommended).
- Install pgvector on Windows by copying the precompiled `vector.dll` into your PostgreSQL `lib` directory, and copying the `vector.control` and `.sql` script files into the `share/extension` directory.

### 5. Start Ollama and Pull the Model
- Download and launch Ollama from [ollama.com](https://ollama.com).
- Pull the model in your terminal:
  ```powershell
  ollama pull qwen2.5:1.5b
  ```

---

## 8. PostgreSQL Database Setup
- **No Database Copying Required**: The database itself is **not** stored in Git. You do not need to copy any files or vector caches from another computer.
- **Schema Re-creation**: A fresh machine creates its own empty database. Calling `init_db()` dynamically creates the schemas and pgvector extension from scratch on your local PostgreSQL database server.
- **Configuration Steps**:
  1. Open psql or pgAdmin and run:
     ```sql
     CREATE DATABASE tech_support_db;
     ```
  2. The application will connect using the credentials specified in your `.env` file's `DATABASE_URL` parameter.
  3. Running the database initializer compiles the schemas:
     ```powershell
     python -c "from database.connection import init_db; init_db()"
     ```

---

## 9. Environment Configuration
Create a `.env` file in the root project workspace to configure local settings:

```env
# FastAPI Service Config
API_HOST=127.0.0.1
API_PORT=8000
DEBUG=True

# Database Connections
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/tech_support_db
DB_ECHO=False

# Ollama LLM Service Settings
OLLAMA_BASE_URL=http://127.0.0.1:11434
LLM_MODEL=qwen2.5:1.5b
LLM_TEMPERATURE=0.0

# Sentence Transformers Embedding Configuration
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# EasyOCR and Pluggable Vision AI settings
OCR_USE_GPU=False
VISION_MODEL_NAME=stub-vision-v1
```

---

## 10. Knowledge Base / Document Ingestion
The raw Spintly API documentation is pre-loaded in the repository under the `data/raw/` directory:
- `data/raw/Spintly API guide.pdf`
- `data/raw/type 1.postman_collection.json`

To generate embeddings and index these documents on a fresh machine:
```powershell
python scripts/ingest.py data/raw/ --force
```
This script reads the raw files, parses endpoint definitions and Postman folders, chunks the text, runs them through the local `all-MiniLM-L6-v2` embedding model, and saves the vectors into the local `tech_support_db` database chunks.

---

## 11. Running the Application

Ensure PostgreSQL and the Ollama daemon are running on the system, then launch the services:

### 1. Start the FastAPI Backend
```powershell
python main.py
```
* The backend server starts on `http://127.0.0.1:8000`.
* Interactive API Documentation (Swagger) is available at: `http://127.0.0.1:8000/docs`.

### 2. Start the Streamlit Web UI Dashboard
Open a new PowerShell terminal, activate your virtual environment, and run:
```powershell
streamlit run dashboard/app.py
```
* Access the web chatbot interface at: `http://localhost:8501`.

---

## 12. How to Verify the System
Confirm each system tier is active before starting a demo:
- **PostgreSQL Connectivity**: Verify that the database service is running and pgvector is initialized.
- **Ollama Reachability**: Open `http://127.0.0.1:11434` in your browser. It should display "Ollama is running".
- **Backend API Health Check**: Visit `http://127.0.0.1:8000/api/v1/health`. It should return a JSON response showing `"status": "ok"`, `"database_connected": true`, and `"ollama_reachable": true`.
- **Streamlit Panel Indicators**: Check the **⚙️ Control Center** sidebar in the Streamlit web dashboard. The health badges for **API**, **DB**, and **LLM** must all display green **Active** status.
- **RAG Verification Tests**: Run `python -m pytest` in the terminal to verify the 39-test integration suite executes successfully.

---

## 13. Google Colab / T4 GPU Setup
The `RAG_Chatbot_Colab.ipynb` notebook provides a complete GPU-accelerated cloud runtime:
1. Open the notebook in Google Colab and set the hardware accelerator to **T4 GPU** (Runtime > Change runtime type > T4 GPU).
2. Run Steps 1–10 to sync the repository, compile `pgvector`, start a local PostgreSQL server, start the backend API, and expose the Streamlit UI via a secure Ngrok public tunnel.
3. Run **Step 11: Run Chatbot Verification Benchmark**.
   - Step 11 operates entirely in-process to prevent subprocess initialization overhead.
   - It performs an untimed warm-up query to compile PyTorch graphs.
   - It polls the Ollama `/api/ps` endpoint to verify `qwen2.5:1.5b` is resident in the GPU VRAM.
   - It executes Query 1 (`"Steps to create a new user"`) under a strict timer, validating that the latency stays below the performance threshold.
   - It executes Queries 2–9 in a fast loop using the same hot pipeline instance.

---

## 14. Benchmark Results
The verified performance baseline measured on a Google Colab T4 GPU instance is:

- **Warm-up**: Successful (PyTorch compiled and Ollama loaded in VRAM).
- **Query 1 latency**: **~3.50 seconds** (10-second requirement: **PASS**).
- **Queries 2–9 latencies**: All consistently between **1.30 and 3.20 seconds**.
- **Ollama VRAM residency**: Verified (`size_vram > 0` and size percentage = 100%).

*Note: These latencies represent execution on a warm GPU-accelerated environment. Local execution on CPU will yield longer response times depending on system hardware.*

---

## 15. Example Questions
You can ask the chatbot standard integration questions based on the Spintly API docs:
- *"Steps to create a new user"*
- *"How do I obtain an OAuth access token?"*
- *"How do I create a meeting?"*
- *"How do I get access points?"*
- *"How do I grant access to an access point?"*

---

## 16. Troubleshooting
- **Database Connection Failure**: Check that your PostgreSQL service is running. Validate your credentials in `.env`'s `DATABASE_URL`.
- **pgvector Extension Missing**: If you get a relation error, make sure you ran `init_db()` and that `vector.dll` was copied to PostgreSQL's extensions directory.
- **Ollama Unreachable**: Ensure the Ollama tray application is running. Check `http://127.0.0.1:11434` in your browser.
- **Hanging Query 1 on Local CPU**: The first request locally takes longer (60–90 seconds) as the model loads into RAM. Subsequent requests will run faster once loaded.
- **Backend Port Already in Use**: If port 8000 is occupied, run `fuser -k 8000/tcp` (on Linux) or stop the conflicting process via Task Manager (on Windows).

---

## 17. Important Portability Note
- **Independent Windows Support**: This codebase is 100% independent of the author's local PC.
- **Fresh Database Creation**: A fresh PC requires only compiling/installing the `pgvector` extension and executing `init_db()`. No database records need to be copied.
- **Cached Weights**: SentenceTransformers will automatically fetch and cache weights on first startup.
- **Model Acquisition**: Running `ollama pull qwen2.5:1.5b` fetches the LLM parameters automatically.
- **Windows pgvector**: Windows precompiled binaries can be copied directly to local PostgreSQL installation folders to load extension capabilities.

---

## 18. GitHub / Repository Usage
- When committing changes, verify that your local `.env` file is ignored (which is handled by default by `.gitignore`).
- Ensure no real credentials or Ngrok access tokens are hard-coded in the notebook before pushing to the repository.

---

## 19. Final Quick Start
For experienced users setting up on a fresh machine:

```powershell
# 1. Clone & Set up Environment
git clone https://github.com/AtharvaNaringrekar/rag-chatbot.git
cd rag-chatbot
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Pull Qwen LLM
ollama pull qwen2.5:1.5b

# 3. Create DB & Ingest
python -c "from database.connection import init_db; init_db()"
python scripts/ingest.py data/raw/ --force

# 4. Start Applications
python main.py
# (In a separate terminal)
streamlit run dashboard/app.py
```
