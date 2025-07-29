import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from rag_workflow.retriever import BM25Retriever
from rag_workflow.utils import parse_nougat_markdown
from rag_workflow.workflow import RAGWorkflow, IngestCompleteEvent, StopEvent

@pytest.fixture
def sample_nodes():
    """Fixture to create sample TextNodes."""
    nodes = [
        TextNode(id_="node1", text="This is a test document about machine learning.", metadata={"file_name": "test.pdf", "page_number": 1}),
        TextNode(id_="node2", text="Deep learning models require large datasets.", metadata={"file_name": "test.pdf", "page_number": 2}),
    ]
    return nodes

@pytest.mark.asyncio
async def test_bm25_retriever(sample_nodes):
    """Test BM25Retriever functionality."""
    retriever = BM25Retriever(nodes=sample_nodes, k=2)
    query_bundle = QueryBundle(query_str="machine learning")
    results = await retriever._aretrieve(query_bundle)
    
    assert isinstance(results, list)
    assert len(results) <= 2
    assert all(isinstance(result, NodeWithScore) for result in results)
    if results:
        assert "machine learning" in results[0].node.text.lower()

def test_parse_nougat_markdown():
    """Test parsing of Nougat markdown output."""
    md_content = """# Page 1
This is a sample paragraph with math $$E=mc^2$$.
## Page 2
Another paragraph with no math."""
    image_analysis = {1: [{"path": "test_image.jpg", "description": "A sample figure"}]}
    filename = "test.pdf"
    
    chunks = parse_nougat_markdown(md_content, image_analysis, filename)
    
    assert len(chunks) == 2
    assert chunks[0][1]["file_name"] == "test.pdf"
    assert chunks[0][1]["page_number"] == 1
    assert chunks[0][1]["contains_math"] is True
    assert chunks[0][1]["latex_formulas"] == ["$$E=mc^2$$"]
    assert chunks[0][1]["llava_descriptions"] == ["A sample figure"]
    assert chunks[0][1]["source_type"] == "text_with_visual"
    assert chunks[1][1]["page_number"] == 2
    assert chunks[1][1]["contains_math"] is False
    assert chunks[1][1]["llava_descriptions"] == []

@pytest.mark.asyncio
async def test_workflow_ingest_no_directory(mocker):
    """Test ingest step with invalid directory."""
    workflow = RAGWorkflow()
    event = StartEvent(dirname="nonexistent_dir")
    ctx = Context()
    result = await workflow.ingest(ctx, event)
    
    assert isinstance(result, StopEvent)
    assert result.result == "Invalid directory"

@pytest.mark.asyncio
async def test_workflow_ingest_empty_pdf(mocker, tmp_path):
    """Test ingest step with empty PDF directory."""
    mocker.patch("rag_workflow.utils.run_nougat_on_pdf", AsyncMock(return_value=None))
    dirname = tmp_path / "data" / "arxiv_papers"
    dirname.mkdir(parents=True)
    workflow = RAGWorkflow()
    event = StartEvent(dirname=str(dirname), query="test query")
    ctx = Context()
    result = await workflow.ingest(ctx, event)
    
    assert isinstance(result, StopEvent)
    assert result.result == "No content extracted from PDFs."