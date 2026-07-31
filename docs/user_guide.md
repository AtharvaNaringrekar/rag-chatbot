# User & Operations Guide

Welcome to the AI Technical Support Assistant. This guide explains how to operate, configure, and use the chatbot dashboard to ingest documents and resolve technical support tickets using screenshot diagnostics.

---

## 🛠️ Step-by-Step Installation

### 1. Set Up Local Services
* **PostgreSQL Database**: Ensure PostgreSQL is running on port 5432 and has the `pgvector` extension installed. Create a database called `tech_support_db`.
* **Ollama Daemon**: Install Ollama and pull the default model:
  ```bash
  ollama pull phi3:mini
  ```

### 2. Install Project Dependencies
Run in your shell terminal:
```bash
# Create and activate python virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Local Startup
Boot the FastAPI Backend server first:
```bash
python main.py
```
This initializes the database schemas. In a second terminal, activate the virtual environment and boot the Streamlit frontend client:
```bash
streamlit run dashboard/app.py
```
Your web portal is now live at: **http://localhost:8501**

---

## 📄 Ingesting Technical Documentation

The assistant can only answer queries using documents ingested into its PostgreSQL vector database.

### Supported File Formats:
1. **API Specifications**:
   * **OpenAPI Swagger (JSON or YAML)**: Resolves `$ref` schema definitions and maps HTTP methods and endpoint paths.
   * **Postman Collections (JSON)**: Maps request bodies, workflows, folders, headers, and query parameters.
2. **Support Manuals**:
   * **PDF**: Extracts page numbers for source citation.
   * **DOCX (Word)**: Preserves headings, tables, and document sections.
   * **TXT (Text)**: Normalizes raw log files and guidebooks.

### Ingestion Steps (CLI admin command):
1. Collect your manuals or specifications in a local directory or note their paths.
2. Ingest a single spec or guidebook:
   ```bash
   python scripts/ingest.py --file path/to/openapi_spec.json
   ```
3. Ingest an entire folder of manuals recursively:
   ```bash
   python scripts/ingest.py --dir path/to/manuals_folder/
   ```

---

## 💬 Querying the Support Assistant

### 1. Document-Grounded Chat
Once documentation is uploaded, type support questions in the chat box, e.g.:
* *"What parameters are required for the /checkout endpoint?"*
* *"What headers do I need to send for authentication?"*

The model will search pgvector, summarize the exact answers, and output cited source documents in an expandable drawer under the response.

### 2. Hallucination Protection
If you ask a question that is *not* answered inside the uploaded documentation (e.g., *"What is the CEO's email address?"*), the model will refuse to guess and output the standard fallback:
> **"I couldn't find that information in the uploaded documentation."**

---

## 🖼️ Screenshot Diagnostics (OCR + Vision)

If developers or users send you screenshots of terminal errors, VS Code debug logs, or Postman failures:

1. Locate the **Screenshot Diagnostic Center** in the sidebar.
2. Upload the screenshot (PNG or JPEG). A preview will render.
3. Type a query in the chat box (e.g. *"What is the fix for this issue?"*) and press enter.
4. The assistant will:
   - Extract raw traceback texts from the image using EasyOCR.
   - Classify the environment layout (e.g. Command Line Terminal vs. IDE code workspace).
   - Retrieve relevant guidelines from pgvector.
   - Formulate a diagnostic step-by-step resolution answer.
5. Review the transcribed logs and layout classes in the **Screenshot Extraction Details** drawer below the message.
