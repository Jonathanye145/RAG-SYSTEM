# 🚀 RAG Workflow: Unleash the Power of Intelligent Document Search

Welcome to the **RAG Workflow Project**! 📚 Transform how you interact with academic papers and PDFs with this cutting-edge Retrieval-Augmented Generation (RAG) system. Powered by advanced AI, it seamlessly processes, indexes, and queries complex documents, delivering precise and contextually rich answers. Whether you're a researcher, student, or knowledge enthusiast, this tool supercharges your ability to extract insights from arXiv papers and beyond! 🌟

## ✨ Why Choose RAG Workflow?

- **Smart PDF Processing**: 📄 Convert PDFs into structured markdown with Nougat, capturing text, LaTeX, and visuals effortlessly.
- **Multimodal Understanding**: 🖼️ Integrate image analysis (via LLaVA placeholder) for tables, charts, and figures.
- **Hierarchical Indexing with RAPTOR**: 🗂️ Organize content intelligently with multi-level summarization for deeper insights.
- **Flexible Retrieval Strategies**: 🔍 Choose from Hybrid, HyDE, or Step-Back retrieval to match your query needs.
- **Human-in-the-Loop Feedback**: 🤝 Refine results interactively to ensure pinpoint accuracy.
- **Validated Answers**: ✅ LLM-driven validation ensures responses are grounded and reliable.

## 📁 Project Structure

The project follows a standard GitHub Python project layout for clarity and maintainability:

```
rag_workflow/
├── src/                           # Source code directory
│   └── rag_workflow/             # Python package for the RAG workflow
│       ├── __init__.py           # Package initialization
│       ├── config.py             # Configuration constants
│       ├── utils.py              # Helper functions for PDF processing
│       ├── retriever.py          # BM25 Retriever implementation
│       ├── workflow.py           # Core RAG workflow logic
│       ├── main.py               # CLI entry point
│       └── api.py                # FastAPI web interface
├── tests/                        # Unit tests
│   └── test_workflow.py          # Tests for workflow components
├── docs/                         # Documentation files
│   └── api.md                    # API documentation
├── data/                         # Data storage for PDFs and outputs
│   ├── arxiv_papers/            # Input directory for PDF files
│   ├── index_storage_raptor_mm/ # ChromaDB index storage
│   ├── nougat_output/           # Nougat markdown outputs (.mmd files)
│   └── image_output/            # Extracted images from PDFs
├── requirements.txt              # Project dependencies
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose for multi-container setup
├── LICENSE                       # MIT License
├── README.md                     # Project documentation (this file)
├── create_folder_structure.py    # Script to create folder structure
```

### Notes on Structure
- **`src/rag_workflow/`**: Contains the core Python package, importable as `rag_workflow`.
- **`tests/`**: Unit tests to ensure code reliability.
- **`docs/`**: API and usage documentation.
- **`data/`**: Centralizes all data-related directories.
- **`requirements.txt`**, **`Dockerfile`**, **`docker-compose.yml`**: Support dependency management and deployment.
- **`LICENSE`**: MIT License for open-source usage.

## ⚙️ Configuration

Customize the RAG Workflow via `src/rag_workflow/config.py`:

- **PDF_DIR**: Input PDFs (`./data/arxiv_papers`).
- **PERSIST_DIR**: ChromaDB index storage (`./data/index_storage_raptor_mm`).
- **NOUGAT_OUTPUT_DIR**: Nougat markdown outputs (`./data/nougat_output`).
- **IMAGE_OUTPUT_DIR**: Extracted images (`./data/image_output`).
- **RAPTOR_MAX_LEVELS**: Hierarchical levels for RAPTOR indexing (default: 3).
- **CHUNK_SIZE_RAPTOR_BASE**: Text chunk size (default: 128).
- **RETRIEVAL_TOP_K**: Documents retrieved per query (default: 7).
- **FUSION_RETRIEVAL_TOP_K**: Fusion strategy documents (default: 10).
- **RERANK_TOP_N**: Top documents after reranking (default: 5).
- **EMBEDDING_MODEL_NAME**: Embedding model (`BAAI/bge-large-en`).

Modify these in `config.py` to optimize performance or adapt to your dataset.

## 🛠️ How It Works

1. **PDF Ingestion** 📥:
   - PDFs in `data/arxiv_papers/` are processed using Nougat to extract text and LaTeX.
   - Images are extracted with PyMuPDF, with placeholder LLaVA descriptions for visuals.
   - Content is parsed into chunks with metadata (e.g., page numbers, LaTeX, visuals).

2. **RAPTOR Indexing** 🗄️:
   - Chunks are hierarchically organized using RAPTOR.
   - Text is split, embedded using `BAAI/bge-large-en`, and clustered into summaries.
   - Indexes are stored in ChromaDB at `data/index_storage_raptor_mm/`.

3. **Retrieval** 🔎:
   - Choose a strategy:
     - **Hybrid**: Combines vector and BM25 retrieval with Reciprocal Rank Fusion.
     - **HyDE**: Generates hypothetical documents for semantic matching.
     - **Step-Back**: Retrieves broader context with a generalized query.
   - Retrieve top-k relevant documents.

4. **Human Feedback** 🤲:
   - Review and refine retrieved documents interactively.

5. **Reranking & Synthesis** 📝:
   - Rerank documents using an LLM for relevance.
   - Synthesize a validated answer grounded in the context.

## 📦 Setup Instructions

Get started in minutes! 🚀

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/rag_workflow.git
   cd rag_workflow
   ```

2. **Create Folder Structure**:
   ```bash
   python create_folder_structure.py
   ```

3. **Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Install Nougat**:
   Follow the [Nougat installation guide](https://github.com/facebookresearch/nougat).

5. **Setup Ollama**:
   Install [Ollama](https://ollama.ai/) and pull the model:
   ```bash
   ollama pull deepseek-r1:14b
   ```

6. **Prepare PDFs**:
   Place PDFs in `data/arxiv_papers/`.

7. **Run Locally (CLI)**:
   ```bash
   python src/rag_workflow/main.py
   ```

## 🛳️ Deployment Instructions

Deploy the RAG Workflow as a containerized web service using Docker and FastAPI.

### Prerequisites
- **Docker**: Install [Docker](https://docs.docker.com/get-docker/).
- **Docker Compose**: Included with Docker Desktop or install separately.
- **Hardware**: Ensure sufficient CPU/GPU and memory (at least 16GB RAM recommended for Ollama and Nougat).
- **NVIDIA GPU (optional)**: Install `nvidia-container-toolkit` for GPU support.

### Steps
1. **Set Environment Variables**:
   ```bash
   export RAG_API_KEY=your-secure-api-key
   ```

2. **Build and Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```
   This starts:
   - The RAG Workflow service with a FastAPI endpoint at `http://localhost:8000`.
   - An Ollama service running `deepseek-r1:14b`.

3. **Access the API**:
   - **Health Check**:
     ```bash
     curl -H "X-API-Key: your-secure-api-key" http://localhost:8000/health
     ```
   - **Query**:
     ```bash
     curl -X POST -H "X-API-Key: your-secure-api-key" -H "Content-Type: application/json" -d '{"query": "What is quantum computing?", "strategy": "hybrid"}' http://localhost:8000/query
     ```
   - **Status**:
     ```bash
     curl -H "X-API-Key: your-secure-api-key" http://localhost:8000/status
     ```

4. **Stop the Services**:
   ```bash
   docker-compose down
   ```

### Additional Deployment Notes
- **Scalability**: Use `gunicorn` with multiple workers (configured in `Dockerfile`) for handling concurrent requests.
- **Ollama Configuration**: Ensure Ollama is accessible at `http://ollama:11434` within the Docker network.
- **Data Persistence**: The `data/` directory is mounted as a volume for persistent storage.
- **CPU-Only Deployment**: Comment out the GPU `deploy` sections in `docker-compose.yml` and use the CPU base image in `Dockerfile`.

## 🧪 Running Tests

Run unit tests to verify functionality:
```bash
pytest tests/
```

## 🎮 Usage

### CLI
1. Run `src/rag_workflow/main.py`.
2. Enter a query (e.g., "Latest advancements in quantum computing") or `exit`.
3. Select a retrieval strategy (`hybrid`, `hyde`, `step_back`).
4. Review documents, provide feedback, and receive a validated answer.

### API
Use the FastAPI endpoint:
```json
POST /query
{
  "query": "What is quantum computing?",
  "strategy": "hybrid"
}
```

## 🛠️ Project Components

- **`src/rag_workflow/config.py`** ⚙️: Configuration settings.
- **`src/rag_workflow/utils.py`** 🧰: PDF processing and parsing utilities.
- **`src/rag_workflow/retriever.py`** 🔍: