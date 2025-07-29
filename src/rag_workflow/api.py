from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import asyncio
from typing import Optional
from .workflow import RAGWorkflow
from .config import PDF_DIR
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Workflow API",
    description="API for querying academic papers using a Retrieval-Augmented Generation workflow.",
    version="1.0.0"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API Key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
VALID_API_KEY = os.getenv("RAG_API_KEY", "default-api-key")  # Set in environment or default

async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if api_key != VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    return api_key

# Request and Response Models
class QueryRequest(BaseModel):
    query: str
    strategy: str = "hybrid"

class QueryResponse(BaseModel):
    result: str
    status: str
    query: str
    strategy: str

class HealthResponse(BaseModel):
    status: str
    ollama_available: bool

# Initialize workflow globally for reuse
workflow = RAGWorkflow(timeout=900, verbose=True)

@app.get("/health", response_model=HealthResponse)
@limiter.limit("10/minute")
async def health_check(request: Request):
    """Check the health of the API and Ollama connection."""
    try:
        # Test Ollama connection
        ollama_response = await workflow.llm.acomplete("Respond with single word: OK")
        ollama_available = ollama_response.text.strip() == "OK"
        return HealthResponse(status="healthy", ollama_available=ollama_available)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="unhealthy", ollama_available=False)

@app.post("/query", response_model=QueryResponse)
@limiter.limit("5/minute")
async def query_documents(request: QueryRequest, api_key: str = Depends(verify_api_key)):
    """Submit a query to the RAG workflow."""
    if request.strategy not in ["hybrid", "hyde", "step_back"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid strategy. Choose 'hybrid', 'hyde', or 'step_back'."
        )
    try:
        logger.info(f"Processing query: '{request.query}' with strategy: '{request.strategy}'")
        final_event = await workflow.run(
            query=request.query,
            dirname=PDF_DIR,
            retrieval_strategy=request.strategy
        )
        if isinstance(final_event, StopEvent):
            logger.info(f"Query completed with result: {final_event.result}")
            return QueryResponse(
                result=final_event.result,
                status="success",
                query=request.query,
                strategy=request.strategy
            )
        else:
            logger.error(f"Workflow did not complete successfully for query: {request.query}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Workflow did not complete successfully."
            )
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/status")
@limiter.limit("10/minute")
async def get_status(request: Request, api_key: str = Depends(verify_api_key)):
    """Get the status of the RAG workflow system."""
    try:
        pdf_count = len([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])
        index_dir_exists = os.path.exists(os.path.join(PDF_DIR, "index_storage_raptor_mm"))
        return {
            "status": "operational",
            "pdf_count": pdf_count,
            "index_available": index_dir_exists,
            "api_version": app.version
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking status: {str(e)}"
        )