import os

# Configuration Constants
PDF_DIR = "./data/arxiv_papers"
PERSIST_DIR = os.path.join(PDF_DIR, "index_storage_raptor_mm")
NOUGAT_OUTPUT_DIR = os.path.join(PDF_DIR, "nougat_output")
IMAGE_OUTPUT_DIR = os.path.join(PDF_DIR, "image_output")
RAPTOR_MAX_LEVELS = 3
CHUNK_SIZE_RAPTOR_BASE = 128
RETRIEVAL_TOP_K = 7
FUSION_RETRIEVAL_TOP_K = 10
RERANK_TOP_N = 5
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"