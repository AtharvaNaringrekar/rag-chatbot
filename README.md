# AI Technical Support Assistant using RAG + Vision AI
### *Internship Project at Spintly*

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql)](https://www.postgresql.org)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A5%97%20Hugging%20Face-Transformers-FFD21E?style=for-the-badge)](https://huggingface.co)
[![Ollama](https://img.shields.io/badge/Ollama-Run%20Local-black?style=for-the-badge)](https://ollama.com)

An enterprise-grade, production-quality AI Chatbot behaving like a Technical Support Engineer. The assistant helps developers, testers, and clients troubleshoot technical issues and understand APIs by answering queries and analyzing error screenshots **strictly** using uploaded company documentation.

Built with **Clean Architecture** principles in pure Python (no LangChain wrapping overhead) to ensure modularity, high performance, and easily swappable component layers.

---

## ⚡ Quick Start (Local Setup)
If you already have PostgreSQL, pgvector, and Ollama installed, get the project running locally in 4 simple commands:
```bash
# 1. Install package dependencies
pip install -r requirements.txt

# 2. Pull Qwen2.5 LLM
ollama pull qwen2.5:1.5b

# 3. Index raw technical documentation into PostgreSQL
python scripts/ingest.py data/raw/ --force

# 4. Start backend & UI (Run main.py, and in a separate terminal run Streamlit)
python main.py
streamlit run dashboard/app.py
```

---

## 🎓 Reviewer & Faculty Evaluation Guide
This project supports two execution modes depending on your hardware availability and grading preferences.

### Method 1: Local VS Code (Recommended for Local Evaluation)
* **When to use**: If you want to evaluate the project's performance offline on a local host machine with at least 8GB of system RAM.
* **LLM Engine**: Runs via **Ollama** (`qwen2.5:1.5b`).
* **Pros**: Low memory footprint, runs entirely offline, and allows debug step-throughs inside VS Code.

### Method 2: Google Colab (Recommended for Fast GPU Testing)
* **When to use**: If your local machine lacks GPU resources or you prefer not to install PostgreSQL and Ollama on your system.
* **LLM Engine**: Runs directly in GPU VRAM via **Hugging Face Transformers** (`Qwen/Qwen2.5-1.5B-Instruct` in `float16` precision).
* **Pros**: Fast GPU inference (T4 GPU acceleration), zero local system dependencies, and public sharing capability via ngrok.

---

## 📖 Table of Contents
1. [Project Description](#-project-description)
2. [Features](#-features)
3. [Tech Stack](#-tech-stack)
4. [Project Architecture](#-project-architecture)
5. [Folder Structure](#-folder-structure)
6. [RAG Workflow](#-rag-workflow)
7. [Local Installation (VS Code Setup Order)](#-local-installation-vs-code-setup-order)
8. [Google Colab Setup](#-google-colab-setup)
9. [Configuration](#-configuration)
10. [Supported Document Types](#-supported-document-types)
11. [Document Ingestion](#-document-ingestion)
12. [Running the Project](#-running-the-project-endpoints)
13. [Running Tests](#-running-tests)
14. [Switching Models](#-switching-models)
15. [Switching Database](#-switching-database)
16. [Deployment](#-deployment)
17. [Troubleshooting](#-troubleshooting)
18. [Future Enhancements](#-future-enhancements)
19. [Disclaimer](#-disclaimer)
20. [Acknowledgement](#-acknowledgement)
21. [Author](#-author)

---

## 📝 Project Description

### Overview
This project was developed during an internship at Spintly. It is an **AI Technical Support Assistant** built using Retrieval-Augmented Generation (RAG) to assist developers, testers, support engineers, and clients by answering integration questions using authorized technical documentation. It supports OCR-based troubleshooting of error screenshots, retrieving relevant reference details from PostgreSQL with the `pgvector` extension, and generating grounded, context-bound answers using local and cloud Large Language Models.

### Problem Statement
Technical support engineers and integration developers spend countless hours parsing dense documentation, testing API requests, and debugging error logs. Off-the-shelf LLMs cannot resolve these issues because they lack access to private corporate documentation and hallucinate non-existent endpoints, parameters, and payloads.

### Why RAG (Retrieval-Augmented Generation) is Used
Instead of fine-tuning models (which is expensive and prone to knowledge cutoff), RAG embeds company documentation into a high-dimensional vector space. When a query is received, the system extracts relevant passages using vector similarity search, injects them into a strict system template, and uses the LLM purely as a reasoning compiler to synthesize a precise response.

### Main Objective
Provide a self-hosted, scalable, and zero-hallucination RAG chatbot that can be deployed locally (for offline use) or on cloud GPU sandboxes (like Google Colab) to accelerate technical support workflows.

---

## ✨ Features

- **Retrieval-Augmented Generation (RAG)**: Connects document matching with generative answers.
- **Strict Grounding Guardrails**: Acts as an AI Technical Support Assistant trained on authorized technical documentation, refusing to answer queries unsupported by reference contexts.
- **PostgreSQL + pgvector Vector Search**: Utilizes cosine distance similarity queries to fetch the most relevant context blocks.
- **FastAPI Backend**: Provides high-performance, asynchronous REST API boundaries (`/ingest`, `/chat`, `/health`).
- **Streamlit Dashboard**: A user-friendly, responsive chat GUI with an upload composer.
- **OCR Support**: Transcribes logs and error text from image attachments using `EasyOCR`.
- **Vision Support**: Pluggable vision engine wrapper prepared for image layout analysis.
- **API Spec Ingestion**: Native parsers for Swagger/OpenAPI JSON/YAML files and Postman JSON Collections.
- **Document Ingestion**: Seamless ingestion of standard manual formats (.pdf, .docx, .txt).
- **Source Citations**: Returns exact document names and metadata associated with matching context chunks.
- **Modular Onion Architecture**: Built with plug-and-play interfaces to swap LLMs, embedders, databases, or parsers.

---

## 🛠️ Tech Stack

* **Programming Language**: Python 3.10+
* **Backend Framework**: FastAPI (ASGI server)
* **Frontend Web Dashboard**: Streamlit
* **Database**: PostgreSQL with `pgvector` (Vector similarity extension)
* **Embeddings Model**: Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
* **OCR Engine**: `EasyOCR`
* **Local Inference LLM**: Ollama (`qwen2.5:1.5b`)
* **Google Colab Inference LLM**: Hugging Face Transformers (`Qwen/Qwen2.5-1.5B-Instruct` in `float16` precision)
* **Libraries & Packages**:
  * ORM: `SQLAlchemy`
  * Drivers: `psycopg2-binary`
  * Text Extraction: `pypdf`, `python-docx`
  * Parsing: `PyYAML`
  * Config Management: `pydantic-settings`

---

## 📐 Project Architecture

The codebase follows **Clean Architecture** patterns, ensuring that domain rules remain isolated from external dependencies.

```mermaid
graph TD
    subgraph Delivery Layer [Delivery Layer]
        API[api/router.py - FastAPI Routes]
        UI[dashboard/app.py - Streamlit UI]
    end

    subgraph Pipeline Layer [Pipeline Layer]
        IP[pipeline/ingestion_pipeline.py]
        RP[pipeline/rag_pipeline.py]
    end

    subgraph Services Layer [Services Layer]
        LLM[services/llm/ - Ollama / HF]
        EMB[services/embedding/ - SentenceTransformers]
        VEC[services/vector_store/ - Postgres Vector]
        OCR[services/ocr/ - EasyOCR]
        PAR[services/parser/ - PDF/Docx/OpenAPI/Postman]
    end

    subgraph Core Layer [Core Domain Layer]
        INT[core/interfaces/ - Base Contracts]
        MOD[core/models.py - Domain Models]
        EXC[core/exceptions.py - Exceptions]
    end

    UI -->|HTTP Requests| API
    API -->|DI Instantiation| RP
    IP --> VEC
    IP --> EMB
    RP --> EMB
    RP --> VEC
    RP --> LLM
    RP --> OCR
    LLM -.->|Implements| INT
    EMB -.->|Implements| INT
    VEC -.->|Implements| INT
    OCR -.->|Implements| INT
    PAR -.->|Implements| INT
```

---

## 📂 Folder Structure

```
technical_support_bot/
├── api/                 # FastAPI router mappings, API schemas, and endpoints
├── config/              # Configuration schemas built via pydantic-settings
├── core/                # Core domain models and abstract interface contracts
│   └── interfaces/      # Interface blueprints (ILLMService, IVectorStore, etc.)
├── dashboard/           # Streamlit UI chat dashboard panels
├── database/            # SQLAlchemy database connection pooling and schema definitions
├── data/                # Data storage for document ingestion directories
│   └── raw/             # Root folder scanned for raw documentation files
├── ingestion/           # Document splitter algorithms and ligature normalizers
├── pipeline/            # RAG orchestration loops and strict prompt templates
├── services/            # Swappable concrete implementations of core interfaces
│   ├── embedding/       # sentence-transformers all-MiniLM-L6-v2 embedding adapter
│   ├── llm/             # Ollama client implementations
│   ├── ocr/             # EasyOCR extraction engines
│   ├── parser/          # PDF, DOCX, TXT, OpenAPI, and Postman parser classes
│   ├── vector_store/    # Postgres/pgvector adapter logic
│   └── vision/          # Vision stub layer
├── scripts/             # Python scripting utilities (cli ingestion commands)
├── tests/               # Pytest unit and integration test suites
├── main.py              # FastAPI application server entrypoint
└── RAG_Chatbot_Colab.ipynb # Google Colab deployment and GPU setup notebook
```

---

## 🔄 RAG Workflow

```mermaid
flowchart TD
    A[Raw Technical Documents] --> B(Ingestion Parser)
    B --> C(Smart Text Normalizer)
    C --> D(Recursive Character Splitter)
    D --> E(SentenceTransformer Embedder)
    E --> F[PostgreSQL + pgvector Database]
    
    G[User Query / Image Attachment] --> H{Contains Image?}
    H -->|Yes| I(EasyOCR Text Extraction)
    H -->|No| J(Generate Query Vector)
    I --> J
    J --> K(pgvector Similarity Query)
    K --> L(Secondary Cosine Reranker)
    L --> M(Filter Threshold < 0.25)
    M --> N(Adaptive Context Prompt Builder)
    N --> O(LLM Reasoning Compiler)
    O --> P[Grounded Answer + Citations]
```

1. **Document Ingestion**: Incoming documents are parsed by extension factories, normalized, and split into recursive character chunks.
2. **Vectorization**: Passages are vectorized into 384-dimensional embeddings and written to the database.
3. **Query Expansion & OCR**: Any attached screenshot is processed by `EasyOCR` and appended to the text query.
4. **Vector Retrieval**: A pgvector similarity search fetches matching candidates, which are reranked and filtered using a similarity threshold ($>= 0.25$).
5. **Context Aggregation**: Context blocks are formatted cleanly and compiled into systemic prompts.
6. **LLM Generation**: The model processes the prompt and streams the finalized response.

---

## 🚀 Local Installation (VS Code Setup Order)

Please follow this exact order to configure the system locally:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AtharvaNaringrekar/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment**
   ```bash
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

4. **Install Python Package Requirements**
   ```bash
   pip install -r requirements.txt
   ```

5. **Install PostgreSQL**
   Install PostgreSQL on your host system (v12+ recommended).

6. **Install pgvector Extension**
   Ensure the `pgvector` extension is compiled and loaded on your Postgres database engine.
   * On Windows: Copy the `vector.dll` and vector SQL files into your PostgreSQL directories.
   * On Linux/macOS: Run `make` and `make install` from the [pgvector repository](https://github.com/pgvector/pgvector.git).

7. **Install Ollama**
   Download and install Ollama from [ollama.com](https://ollama.com).

8. **Pull local LLM Model**
   Pull the 1.5B Qwen instruct model:
   ```bash
   ollama pull qwen2.5:1.5b
   ```

9. **Configure Environment (.env)**
   Create a `.env` file in the root workspace folder:
   ```env
   API_HOST=127.0.0.1
   API_PORT=8000
   DEBUG=True
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/tech_support_db
   DB_ECHO=False
   OLLAMA_BASE_URL=http://localhost:11434
   LLM_MODEL=qwen2.5:1.5b
   LLM_TEMPERATURE=0.0
   ```

10. **Initialize the Database & Schema**
    Ensure PostgreSQL is running, then generate schemas:
    ```bash
    python -c "from database.connection import init_db; init_db()"
    ```

11. **Ingest Documentation Manuals**
    ```bash
    python scripts/ingest.py data/raw/ --force
    ```

12. **Start backend FastAPI Server**
    ```bash
    python main.py
    ```

13. **Start Streamlit UI Dashboard**
    In a new terminal shell:
    ```bash
    streamlit run dashboard/app.py
    ```

---

## 📓 Google Colab Setup

For cloud sandbox environments, the setup uses GPU acceleration and loads models directly via Hugging Face instead of local Ollama daemons.

> [!IMPORTANT]
> **Ephemeral Google Colab Runtimes Warning**:
> * Google Colab virtual machine runtimes are **temporary**. When the runtime disconnects or is recycled (due to inactivity), the environment resets.
> * If the runtime resets, you **must rerun the notebook cells** from Step 1 to Step 8 to recreate the local PostgreSQL server and reload the Hugging Face model.
> * Documents uploaded and ingested in the Colab VM are **temporary**. To preserve them, they must be added to the project repository's `data/raw/` directory and committed/pushed separately.
> * **Step 5A is optional** and should only be run if you want to ingest new custom manuals. By default, Step 6 automatically scans the database and ingests the preloaded repository documents from `data/raw/` if the database is empty.

### Steps to Run

1. Open `RAG_Chatbot_Colab.ipynb` in Google Colab.
2. Go to **Runtime -> Change runtime type** and select **T4 GPU**.
3. **Step 1 & 2**: Execute cells to check CUDA status and clone the repository.
4. **Step 3**: Install system packages (PostgreSQL, pgvector, Poppler, Tesseract) and python dependencies.
5. **Step 4**: Start PostgreSQL service, set credentials, enable `vector`, and run database schemas generation.
6. **Step 5A (Optional Ingestion)**: Run **Step 5A** to upload your own PDF/DOCX/TXT files via the browser dialog and index them.
7. **Step 5 & 6**: Load `Qwen/Qwen2.5-1.5B-Instruct` in `float16` precision to GPU memory and start the FastAPI background server.
8. **Step 7 & 8**: Run Streamlit in the background, set up the `pyngrok` tunnel, input your ngrok Auth Token, and open the generated HTTPS link to launch the dashboard.

---

## ⚙️ Configuration

The following application configurations are defined in [config/settings.py](file:///d:/Technical%20support%20bot/config/settings.py) and can be overridden via environment variables or a `.env` file:

| Configuration Key | Current Default Value | Purpose |
| :--- | :--- | :--- |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/tech_support_db` | Connection coordinates for PostgreSQL database. |
| `LLM_MODEL` | `qwen2.5:1.5b` | Target local LLM model name loaded by Ollama. |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Endpoint for the Ollama API daemon. |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Model name for vector embeddings. |
| `DEFAULT_CHUNK_SIZE` | `1000` | Target character count constraint for text splits. |
| `DEFAULT_CHUNK_OVERLAP` | `100` | Overlap character count between text chunks. |
| `OCR_USE_GPU` | `False` | Toggles GPU acceleration for OCR image transcribing. |
| `VISION_MODEL_NAME` | `llava:latest` | Target multimodal model name for image analysis. |
| `API_HOST` | `127.0.0.1` | Local IP binding for uvicorn server. |
| `API_PORT` | `8000` | Local port binding for uvicorn server. |

---

## 📂 Supported Document Types

| Document Category | Format Extensions | Parser File |
| :--- | :--- | :--- |
| **PDF Manuals** | `.pdf` | `services/parser/doc_parsers.py` |
| **Word Documents** | `.docx` | `services/parser/doc_parsers.py` |
| **Plain Text Files**| `.txt` | `services/parser/doc_parsers.py` |
| **OpenAPI / Swagger**| `.json`, `.yaml`, `.yml` | `services/parser/openapi.py` |
| **Postman Collections**| `.json` | `services/parser/postman.py` |

---

## 📥 Document Ingestion

The CLI ingestion script `scripts/ingest.py` parses documents and indices them.

### Examples

<details>
<summary><b>1. Ingest a Single Manual File</b></summary>

```bash
python scripts/ingest.py data/raw/spintly_auth_guide.pdf
```
</details>

<details>
<summary><b>2. Ingest Multiple Specific Files</b></summary>

```bash
python scripts/ingest.py data/raw/guide1.txt data/raw/guide2.docx
```
</details>

<details>
<summary><b>3. Recursively Ingest a Folder</b></summary>

```bash
python scripts/ingest.py data/raw/
```
</details>

<details>
<summary><b>4. Force Re-indexing & Overwriting Duplicates</b></summary>

By default, the ingester warns on duplicates. Use `--force` to bypass validation prompts:
```bash
python scripts/ingest.py data/raw/ --force
```
</details>

---

## 🚀 Running the Project Endpoints

### 1. Start backend server
```bash
python main.py
```
* Swagger UI Docs: `http://127.0.0.1:8000/docs`
* API Health Check: `http://127.0.0.1:8000/api/v1/health`

### 2. Start frontend dashboard
```bash
streamlit run dashboard/app.py
```
Open `http://localhost:8501` to access the conversational interface.

---

## 🧪 Running Tests

Unit and integration tests are located in `tests/`. Run them using `pytest`:

```bash
# Run all unit and integration tests
pytest

# Test document parsers
pytest tests/test_doc_parsers.py

# Test API endpoints routing
pytest tests/test_api_routes.py

# Run RAG Query flow integration tests
pytest tests/cli_rag_test.py
```

---

## 🔄 Switching Models

The chatbot supports model changes without requiring code modifications:

1. **Local (Ollama)**:
   Open `.env` or `config/settings.py` and modify `LLM_MODEL`:
   ```env
   LLM_MODEL=qwen2.5:3b
   ```
   *No architectural changes are required because the routing layer communicates with services through the abstract `ILLMService` interface.*

2. **Google Colab (Hugging Face)**:
   In `RAG_Chatbot_Colab.ipynb`, edit Step 5 and Step 6 model name overrides:
   ```python
   model_name = "Qwen/Qwen2.5-3B-Instruct"
   ```

---

## 💾 Switching Database

To connect the application to an external or production PostgreSQL instance:
1. Access `.env` or set environment variable `DATABASE_URL`.
2. Update the connection parameters:
   ```env
   DATABASE_URL=postgresql://your_user:your_password@your_host:5432/your_database
   ```
3. Initialize the schema tables on the new server:
   ```python
   python -c "from database.connection import init_db; init_db()"
   ```

---

## 🌐 Deployment

### Local Deployment
* Uses local **Ollama** daemons running on host CPU or local GPU acceleration.
* Database is hosted on localhost PostgreSQL databases.
* Suitable for secure, offline environments.

### Google Colab Deployment
* Bypasses local Ollama setup and loads models in `float16` precision using GPU acceleration.
* Uses local ephemeral PostgreSQL databases inside the VM workspace.
* Accessible from any web browser using secure public ngrok HTTPS proxy tunnels.

---

## 🛠️ Troubleshooting

* **Ollama Connection Refused**:
  Make sure the Ollama application is running in the background. Check if `http://localhost:11434` is accessible in your browser.
* **Database Connection Timed Out**:
  Ensure the PostgreSQL service is active. On local setups, verify credentials in `.env`. On Colab, ensure `service postgresql start` executed successfully.
* **Incompatible Transformers Version in Colab**:
  If model loading throws a `KeyError: 'type'`, rerun the Step 5 package installation block. The cell will upgrade `transformers` to `>=4.41.0` automatically.
* **ngrok Connection Limits**:
  If the public tunnel fails, configure your `NGROK_AUTHTOKEN` in Step 8 to prevent session timeouts.

---

## 🔮 Future Enhancements

* **True Multimodality**: Replace the vision classification stub with active visual transformers (e.g. Qwen-VL) to directly answer visual interface questions.
* **Hybrid Search Retrieval**: Integrate keyword-based Sparse Search (BM25) with Dense Vector Search (pgvector) to improve lexical query accuracy.
* **Dynamic Chunker**: Implement semantic chunking models that partition documents along thematic boundaries instead of static character counts.

---

## ⚖️ Disclaimer

* This project was developed during an internship at Spintly.
* The repository is intended solely for educational, demonstrative, and portfolio purposes.
* Proprietary company documentation, confidential information, credentials, and internal company assets are **NOT** included in this public repository.
* Any mock documentation or test guides used during development are not distributed with this project.

---

## 🤝 Acknowledgement

This project was developed during an internship at Spintly. We gratefully acknowledge the guidance, support, and feedback provided by the internship mentors and the engineering team at Spintly during the development of this prototype.

---

## 👥 Author

* **Developer**: Atharva Naringrekar (GitHub: [@AtharvaNaringrekar](https://github.com/AtharvaNaringrekar))
* **Project**: AI Technical Support Assistant using RAG + Vision AI
* **Developed As**: Internship Project at Spintly
* **Technology Stack**: FastAPI, Streamlit, PostgreSQL + pgvector, Ollama, Hugging Face, Python
* **Status**: Internship Project – Production Prototype
