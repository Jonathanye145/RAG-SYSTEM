from typing import List
from llama_index.core.retrievers import BaseRetriever, QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode
from rank_bm25 import BM25Okapi
import re
import asyncio

class BM25Retriever(BaseRetriever):
    """Simple BM25 Retriever wrapper."""
    def __init__(self, nodes: List[TextNode], tokenizer=None, k=5):
        self._nodes = nodes
        self._corpus = [node.get_content() if node.get_content() else "" for node in nodes]
        self._tokenizer = tokenizer if tokenizer else lambda s: re.findall(r'\b\w+\b', s.lower())
        if not self._corpus or all(not doc for doc in self._corpus):
            print("Warning: BM25 Corpus empty.")
            self._bm25 = None
        else:
            try:
                tokenized_corpus = [self._tokenizer(doc) for doc in self._corpus]
            except Exception as tok_err:
                print(f"ERROR: BM25 tokenization failed: {tok_err}")
                self._bm25 = None
                return
            if not any(tokenized_corpus):
                print("Warning: BM25 tokenization resulted in empty corpus.")
                self._bm25 = None
            else:
                try:
                    self._bm25 = BM25Okapi(tokenized_corpus)
                except ValueError as ve:
                    print(f"ERROR: BM25Okapi initialization failed (ValueError): {ve}. Corpus sample: {self._corpus[:2]}")
                    self._bm25 = None
                except Exception as bm25_init_err:
                    print(f"ERROR: Failed to initialize BM25: {bm25_init_err}")
                    self._bm25 = None
        self._k = k
        super().__init__()

    def _retrieve(self, query: str) -> List[NodeWithScore]:
        if self._bm25 is None:
            return []
        try:
            tokenized_query = self._tokenizer(query)
        except Exception as tok_err:
            print(f"ERROR: BM25 query tokenization failed: {tok_err}")
            return []
        if not tokenized_query:
            return []
        try:
            doc_scores = self._bm25.get_scores(tokenized_query)
        except Exception as score_err:
            print(f"ERROR: BM25 scoring failed: {score_err}")
            return []

        node_scores = []
        for i, score in enumerate(doc_scores):
            if i < len(self._nodes) and isinstance(score, (float, int)):
                node_scores.append((self._nodes[i], float(score)))

        node_scores.sort(key=lambda item: item[1], reverse=True)
        top_k_nodes = node_scores[:self._k]
        return [NodeWithScore(node=node, score=score) for node, score in top_k_nodes]

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Async retrieval method compatible with LlamaIndex QueryBundle."""
        query_str = query_bundle.query_str
        return await asyncio.to_thread(self._retrieve, query_str)