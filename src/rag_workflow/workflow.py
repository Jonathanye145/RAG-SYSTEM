import asyncio
from typing import List, Dict
from tqdm import tqdm
import traceback
from llama_index.core import (
    VectorStoreIndex, StorageContext, Document, SimpleDirectoryReader,
    VectorStoreIndex, StorageContext, load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.workflow import Context, Workflow, StartEvent, StopEvent, Event, step
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.config import Settings
import chromadb
from .config import *
from .utils import parse_nougat_markdown, extract_images_and_regions, get_llava_description, remove_think_blocks
from .retriever import BM25Retriever

# Define custom events
class IngestCompleteEvent(Event):
    query: str

class RetrieverEvent(Event):
    nodes: list[NodeWithScore]

class RerankEvent(Event):
    nodes: list[NodeWithScore]

class ApprovedEvent(Event):
    nodes: list[NodeWithScore]

class RAGWorkflow(Workflow):
    def __init__(self, *args, llm=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm or Ollama(model="deepseek-r1:14b", request_timeout=300.0)

    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> IngestCompleteEvent | StopEvent | None:
        dirname = ev.get("dirname") or await ctx.get("dirname")
        query = ev.get("query") or await ctx.get("original_query")
        print(f"DEBUG: Ingest start. dir='{dirname}', q='{query}'")
        if not dirname or not os.path.isdir(dirname):
            print(f"ERROR: Invalid dir '{dirname}'")
            return StopEvent(result="Invalid directory")

        pdf_files = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith(".pdf")]
        os.makedirs(PERSIST_DIR, exist_ok=True)
        os.makedirs(NOUGAT_OUTPUT_DIR, exist_ok=True)
        os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

        # Part 1: PDF Processing
        all_base_chunks = []
        print("Starting PDF processing phase (Nougat + LLaVA placeholders)...")
        async def process_single_pdf(pdf_path):
            filename = os.path.basename(pdf_path)
            md_path = await run_nougat_on_pdf(pdf_path, NOUGAT_OUTPUT_DIR)
            if not md_path:
                return filename, None
            img_meta = extract_images_and_regions(pdf_path, IMAGE_OUTPUT_DIR)
            pdf_llava_tasks = []
            items_with_paths = []
            for page, items in img_meta.items():
                for item in items:
                    if item.get('path'):
                        pdf_llava_tasks.append(get_llava_description(item['path']))
                        items_with_paths.append(item)
            try:
                descriptions = await asyncio.gather(*pdf_llava_tasks)
                for item, desc in zip(items_with_paths, descriptions):
                    if desc:
                        item['description'] = desc
            except Exception as llava_gather_err:
                print(f"Warn: Error during LLaVA gathering for {filename}: {llava_gather_err}")
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
            except Exception as read_err:
                print(f"Warn: Failed to read Nougat output {md_path}: {read_err}")
                return filename, None
            pdf_chunks = parse_nougat_markdown(md_content, img_meta, filename)
            return filename, pdf_chunks

        pdf_tasks = [process_single_pdf(pdf_path) for pdf_path in pdf_files]
        for future in tqdm(asyncio.as_completed(pdf_tasks), total=len(pdf_tasks), desc="PDF Process (Nougat+LLaVA)"):
            try:
                filename, result_chunks = await future
                if result_chunks is not None:
                    all_base_chunks.extend(result_chunks)
                    print(f"Successfully processed {filename}, added {len(result_chunks)} chunks.")
                else:
                    print(f"Processing failed for {filename}.")
            except Exception as proc_err:
                print(f"Warn: Exception processing a PDF: {proc_err}")
                traceback.print_exc()
        if not all_base_chunks:
            print("ERROR: No content was extracted from any PDF after processing.")
            return StopEvent(result="No content extracted from PDFs.")

        # Part 2: Initialize Embedding Model
        embeddings_model: BaseEmbedding | None = None
        try:
            print(f"Initializing embedding model '{EMBEDDING_MODEL_NAME}'...")
            embeddings_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
            _ = await embeddings_model.aget_text_embedding("test")
            print("Embedding model initialized successfully.")
        except Exception as embed_err:
            print(f"CRITICAL Error initializing embedding model: {embed_err}")
            traceback.print_exc()
            return StopEvent(result="Failed to initialize embedding model.")

        # Part 3: Index Loading / Building
        index = None
        all_nodes_for_index = []
        storage_context = None
        try:
            print(f"Attempting to load index from {PERSIST_DIR}...")
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(
                storage_context,
                embed_model=embeddings_model
            )
            all_nodes_from_store = list(storage_context.docstore.docs.values())
            if not all_nodes_from_store:
                raise ValueError("Loaded index but docstore is empty. Rebuilding required.")
            print(f"Successfully loaded index with {len(all_nodes_from_store)} nodes.")
            all_nodes_for_index = all_nodes_from_store
        except Exception as load_err:
            print(f"Failed to load index or index inconsistent ({type(load_err).__name__}: {load_err}). Rebuilding index...")
            print("Starting RAPTOR build process using extracted content...")
            all_levels = []
            node_id_counter = 0
            level0 = []
            splitter = SentenceSplitter(chunk_size=CHUNK_SIZE_RAPTOR_BASE, chunk_overlap=CHUNK_SIZE_RAPTOR_BASE//5)
            for text, meta in tqdm(all_base_chunks, desc="Creating L0 Nodes"):
                temp_doc = Document(text=text, metadata=meta)
                chunks = splitter.split_text(text)
                for chunk_idx, chunk in enumerate(chunks):
                    node_id = f"n_{meta.get('file_name', 'unk')}_p{meta.get('page_number', 0)}_{node_id_counter}_c{chunk_idx}"
                    node = TextNode(
                        id_=node_id,
                        text=chunk,
                        metadata={**meta, "level": 0, "child_ids": [], "parent_id": None}
                    )
                    level0.append(node)
                    node_id_counter += 1
            if not level0:
                print("ERROR: No L0 nodes were created during splitting.")
                return StopEvent(result="No L0 nodes created.")
            print(f"Created {len(level0)} L0 nodes.")
            current_level = level0
            all_levels.append(current_level)
            for level in range(RAPTOR_MAX_LEVELS):
                print(f"Processing RAPTOR Level {level + 1}...")
                if len(current_level) <= 1:
                    print(f"Stopping recursion at L{level+1}: Not enough nodes ({len(current_level)}) to cluster/summarize.")
                    break
                print(f"Embedding {len(current_level)} nodes for L{level+1}...")
                texts_to_embed = [n.get_content(metadata_mode="none") for n in current_level]
                try:
                    embeddings = await embeddings_model.aget_text_embedding_batch(
                        texts_to_embed, show_progress=True
                    )
                except Exception as embed_batch_err:
                    print(f"ERROR during L{level+1} embedding: {embed_batch_err}. Stopping RAPTOR recursion.")
                    traceback.print_exc()
                    break
                print("Clustering nodes (Placeholder - Modulo)...")
                num_clusters = max(1, int(len(current_level)**0.5 / 2))
                clusters = {i: [] for i in range(num_clusters)}
                for i, node in enumerate(current_level):
                    clusters[i % num_clusters].append(node)
                print(f"Created {num_clusters} clusters for L{level+1}.")
                next_level = []
                summary_tasks = []
                print(f"Generating summaries for {len(clusters)} clusters at L{level+1}...")
                for cluster_id, nodes_in_cluster in clusters.items():
                    if not nodes_in_cluster:
                        continue
                    ctx_texts = [n.get_content(metadata_mode="none") for n in nodes_in_cluster[:5]]
                    prompt = f"Summarize the core information from these related text snippets concisely:\n{'---'.join(ctx_texts)}\n\nConcise Summary:"
                    summary_tasks.append(
                        asyncio.create_task(
                            self.llm.acomplete(prompt),
                            name=f"Summary_L{level+1}_C{cluster_id}"
                        )
                    )
                summary_results = await asyncio.gather(*summary_tasks, return_exceptions=True)
                cluster_ids_list = list(clusters.keys())
                summary_node_counter = 0
                for i, result_or_exc in tqdm(enumerate(summary_results), total=len(summary_results), desc=f"Processing L{level+1} Summaries"):
                    cluster_id = cluster_ids_list[i]
                    nodes_in_cluster = clusters[cluster_id]
                    if not nodes_in_cluster:
                        continue
                    summary = f"Failed or empty summary for cluster {cluster_id}"
                    if isinstance(result_or_exc, Exception):
                        print(f"Warn: Summarization task failed for cluster {cluster_id}: {result_or_exc}")
                    elif result_or_exc and result_or_exc.text.strip():
                        summary = result_or_exc.text.strip()
                    children_ids = [n.node_id for n in nodes_in_cluster]
                    meta = {
                        "level": level + 1,
                        "child_ids": children_ids,
                        "parent_id": None,
                        "contains_math": any(n.metadata.get("contains_math", False) for n in nodes_in_cluster),
                        "has_visuals": any(n.metadata.get("llava_descriptions") for n in nodes_in_cluster),
                        "source_files": list(set(n.metadata.get("file_name", "unknown") for n in nodes_in_cluster))
                    }
                    s_id = f"s_{level+1}_{cluster_id}_{summary_node_counter}"
                    s_node = TextNode(id_=s_id, text=summary, metadata=meta)
                    next_level.append(s_node)
                    summary_node_counter += 1
                    for node in nodes_in_cluster:
                        node.metadata["parent_id"] = s_id
                if not next_level:
                    print(f"Stopping recursion at L{level+1}: No summary nodes generated.")
                    break
                print(f"Generated {len(next_level)} L{level+1} nodes.")
                current_level = next_level
                all_levels.append(current_level)
            all_nodes_for_index = [node for level_nodes in all_levels for node in level_nodes]
            if not all_nodes_for_index:
                print("ERROR: No nodes were generated by the RAPTOR process.")
                return StopEvent(result="No nodes generated during index build.")
            print(f"RAPTOR process finished. Total nodes generated: {len(all_nodes_for_index)}")
            print(f"Starting final indexing of {len(all_nodes_for_index)} nodes...")
            try:
                print(f"Initializing ChromaDB PersistentClient with explicit settings at: {PERSIST_DIR}")
                client = chromadb.PersistentClient(
                    path=PERSIST_DIR,
                    settings=Settings(
                        is_persistent=True,
                        persist_directory=PERSIST_DIR
                    )
                )
                collection_name = "raptor_mm_collection_v2"
                print(f"Using ChromaDB collection: {collection_name}")
                coll = client.get_or_create_collection(collection_name)
                vs = ChromaVectorStore(chroma_collection=coll)
                storage_context = StorageContext.from_defaults(vector_store=vs)
                print(f"Adding {len(all_nodes_for_index)} nodes to the Docstore...")
                storage_context.docstore.add_documents(all_nodes_for_index, allow_update=True)
                print("Nodes added to Docstore.")
                print("Creating VectorStoreIndex...")
                index = VectorStoreIndex(
                    nodes=[],
                    storage_context=storage_context,
                    embed_model=embeddings_model,
                    show_progress=True
                )
                print("VectorStoreIndex created.")
                print("Persisting index and storage context...")
                storage_context.persist(persist_dir=PERSIST_DIR)
                print("Index rebuilt and persisted successfully.")
            except Exception as chroma_init_err:
                print(f"CRITICAL ERROR Initializing/Using ChromaDB PersistentClient: {chroma_init_err}")
                traceback.print_exc()
                return StopEvent(result="Failed to initialize/use persistent ChromaDB.")
        if not all_nodes_for_index:
            print("ERROR: No nodes available for BM25 retriever initialization.")
            return StopEvent(result="BM25 fail: No nodes available.")
        print("Initializing BM25 retriever...")
        text_nodes_for_bm25 = [node for node in all_nodes_for_index if isinstance(node, TextNode)]
        if not text_nodes_for_bm25:
            print("ERROR: No TextNodes found among indexed nodes for BM25.")
            return StopEvent(result="BM25 fail: No TextNodes found.")
        try:
            bm25_retriever = BM25Retriever(nodes=text_nodes_for_bm25, k=RETRIEVAL_TOP_K)
            await ctx.set("bm25_retriever", bm25_retriever)
            print("BM25 retriever initialized.")
        except Exception as bm25_err:
            print(f"ERROR: Failed to initialize BM25Retriever: {bm25_err}")
            traceback.print_exc()
            return StopEvent(result=f"BM25 initialization failed: {bm25_err}")
        await ctx.set("index", index)
        await ctx.set("original_query", query)
        await ctx.set("dirname", dirname)
        if query:
            print("Ingest step complete. Proceeding to retrieve.")
            return IngestCompleteEvent(query=query)
        else:
            print("Ingest step complete (index loaded/built, no query provided).")
            return StopEvent(result=index)

    @step
    async def retrieve(self, ctx: Context, ev: IngestCompleteEvent) -> RetrieverEvent | StopEvent | None:
        query = ev.query
        index = await ctx.get("index")
        strategy = await ctx.get("retrieval_strategy", "hybrid")
        original_query = await ctx.get("original_query")
        bm25_retriever = await ctx.get("bm25_retriever")
        if not query or not index or not original_query:
            print("ERROR: Retrieve context missing (query/index/original_query).")
            return StopEvent(result="Error: Context missing.")
        if strategy != "hyde" and not bm25_retriever:
            print(f"ERROR: BM25 retriever missing for selected strategy '{strategy}'.")
            return StopEvent(result="Error: BM25 missing.")
        print(f"Retrieving for '{query}' using strategy: '{strategy}'")
        try:
            vector_retriever = index.as_retriever(similarity_top_k=RETRIEVAL_TOP_K)
            final_nodes: List[NodeWithScore] = []
            if strategy == "hyde":
                print("Executing HyDE Strategy (using TransformQueryEngine)...")
                hyde_transform = HyDEQueryTransform(include_original=True, llm=self.llm)
                base_retriever_engine = RetrieverQueryEngine.from_args(vector_retriever)
                hyde_engine = TransformQueryEngine(base_retriever_engine, hyde_transform)
                print(f"HyDE: Generating hypothetical document for query: '{query}'")
                response = await hyde_engine.aquery(query)
                final_nodes = response.source_nodes if response.source_nodes else []
                print(f"HyDE retrieved {len(final_nodes)} source nodes.")
            elif strategy == "step_back":
                print("Executing Step-Back Strategy (using QueryFusionRetriever)...")
                step_back_prompt = f"Generate a more general, high-level question based on the specific query: '{query}'. Output ONLY the general question."
                step_back_resp = await self.llm.acomplete(step_back_prompt)
                step_back_query = step_back_resp.text.strip()
                if not step_back_query or step_back_query.lower() == query.lower():
                    print("Warning: Step-back query generation failed or returned original. Using original query for step-back retrieval.")
                    step_back_query = query
                else:
                    print(f"Generated Step-Back Query: '{step_back_query}'")
                print("Initializing Fusion Retriever for Step-Back...")
                fusion_retriever = QueryFusionRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    similarity_top_k=FUSION_RETRIEVAL_TOP_K,
                    num_queries=1,
                    mode="reciprocal_rerank",
                    use_async=True,
                    verbose=True,
                )
                print("Retrieving (Original Query with Fusion)...")
                fused_orig_nodes = await fusion_retriever.aretrieve(query)
                fused_sb_nodes = []
                if step_back_query != query:
                    print("Retrieving (Step-Back Query with Fusion)...")
                    fused_sb_nodes = await fusion_retriever.aretrieve(step_back_query)
                all_fused_nodes = fused_orig_nodes + fused_sb_nodes
                unique_nodes_dict: Dict[str, NodeWithScore] = {}
                for node_w_score in sorted(all_fused_nodes, key=lambda n: n.score or 0.0, reverse=True):
                    if node_w_score.node.node_id not in unique_nodes_dict:
                        unique_nodes_dict[node_w_score.node.node_id] = node_w_score
                final_nodes = list(unique_nodes_dict.values())
                print(f"Step-Back combined/deduplicated into {len(final_nodes)} nodes.")
            else:
                print("Executing Hybrid Strategy (Vector + BM25 via QueryFusionRetriever)...")
                fusion_retriever = QueryFusionRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    similarity_top_k=FUSION_RETRIEVAL_TOP_K,
                    num_queries=1,
                    mode="reciprocal_rerank",
                    use_async=True,
                    verbose=True,
                    llm=self.llm
                )
                final_nodes = await fusion_retriever.aretrieve(query)
                print(f"QueryFusionRetriever (RRF - Hybrid) retrieved {len(final_nodes)} nodes.")
            if not final_nodes:
                print("Warning: Retrieval returned no nodes.")
            return RetrieverEvent(nodes=final_nodes)
        except Exception as retrieve_err:
            print(f"CRITICAL Retrieve Error: {retrieve_err}")
            traceback.print_exc()
            return StopEvent(result=f"Fail retrieve: {retrieve_err}")

    @step
    async def human_feedback(self, ctx: Context, ev: RetrieverEvent) -> ApprovedEvent | StartEvent | StopEvent:
        original_query = await ctx.get("original_query")
        index = await ctx.get("index")
        dirname = await ctx.get("dirname")
        if not index or not original_query or not dirname:
            print("ERROR: Feedback context missing.")
            return StopEvent(result="Error: Context missing.")
        nodes = ev.nodes
        max_iterations = 3
        iteration = 0
        current_nodes = nodes
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Feedback Iteration {iteration}/{max_iterations} ---")
            print(f"Original Query: '{original_query}'")
            if not current_nodes:
                print("No nodes to review for feedback.")
                retry_choice = input("No documents found/remaining. Restart query process? (y/n): ").strip().lower()
                return StartEvent(dirname=dirname, query=original_query) if retry_choice == 'y' else StopEvent(result="No relevant documents found after feedback.")
            print("Current nodes for review:")
            for i, n in enumerate(current_nodes[:10]):
                print(f"[{i}] Score:{n.score or 0.0:.4f} (ID: {n.node.node_id})\n   {n.node.text[:250].replace(chr(10), ' ')}...\n")
            feedback_input = input(f"\nReview nodes. Options:\n"
                                  f"  'ok', 'yes', 'y'      : Approve current {len(current_nodes)} nodes -> Rerank & Synthesize.\n"
                                  f"  'none', 'restart', 'n': Discard results -> Start a new query.\n"
                                  f"  <your feedback>    : Provide feedback -> Refine search & review again.\n"
                                  f"Your choice: ").strip()
            feedback_lower = feedback_input.lower()
            if not feedback_input or feedback_lower in ['ok', 'good', 'yes', 'approve', 'y']:
                print(f"User approved {len(current_nodes)} nodes.")
                return ApprovedEvent(nodes=current_nodes)
            elif feedback_lower in ['none', 'discard', 'restart', 'n']:
                print("Discarding results and restarting query process...")
                return StartEvent(dirname=dirname)
            else:
                print(f"Interpreting feedback for refinement: '{feedback_input}'")
                interpretation_prompt = (f"Original Query:'{original_query}'\nUser Feedback:'{feedback_input}'\n"
                                        f"Generate up to 3 improved search queries based ONLY on the feedback, "
                                        f"focusing on what the user wants instead or additionally. "
                                        f"Return ONLY the queries, one per line.")
                try:
                    interpretation_raw = await self.llm.acomplete(interpretation_prompt)
                    cleaned_output = interpretation_raw.text.strip()
                    potential_queries = [q.strip() for q in cleaned_output.split("\n") if q.strip() and len(q) > 3]
                    if potential_queries:
                        new_queries = potential_queries[:3]
                        print(f"Generated refinement queries: {new_queries}")
                        print("Refining search using generated queries (Vector only, top 3 per query)...")
                        feedback_retriever = index.as_retriever(similarity_top_k=3)
                        new_nodes_list = []
                        refine_tasks = [feedback_retriever.aretrieve(q) for q in new_queries]
                        refine_results = await asyncio.gather(*refine_tasks)
                        for retrieved_nodes in refine_results:
                            new_nodes_list.extend(retrieved_nodes)
                        unique_refined_nodes = {n.node.node_id: n for n in new_nodes_list}
                        refined_nodes_list = list(unique_refined_nodes.values())
                        refined_nodes_list.sort(key=lambda n: n.score or 0.0, reverse=True)
                        print(f"Refined search retrieved {len(refined_nodes_list)} unique nodes.")
                        current_nodes = refined_nodes_list
                        continue
                    else:
                        print("LLM failed to generate refinement queries from feedback. Please provide different feedback, or type 'ok'/'none'.")
                except Exception as llm_err:
                    print(f"Error during feedback interpretation/retrieval: {llm_err}. Please try again, 'ok', or 'none'.")
                    traceback.print_exc()
        print(f"\nMaximum feedback iterations ({max_iterations}) reached.")
        if current_nodes:
            print(f"Proceeding with the last set of {len(current_nodes)} nodes after feedback.")
            return ApprovedEvent(nodes=current_nodes)
        else:
            print("No nodes approved after maximum feedback iterations.")
            return StopEvent(result="Failed to find relevant documents after feedback process.")

    @step
    async def rerank(self, ctx: Context, ev: ApprovedEvent) -> RerankEvent | StopEvent:
        if not ev.nodes:
            print("Rerank: No nodes were approved from the feedback step. Skipping rerank.")
            return RerankEvent(nodes=[])
        query = await ctx.get("original_query")
        if not query:
            print("ERROR: Rerank context missing query.")
            return StopEvent(result="Error: Query missing for rerank.")
        print(f"Reranking {len(ev.nodes)} approved nodes for query: '{query}' (requesting top {RERANK_TOP_N})")
        try:
            reranker = LLMRerank(
                choice_batch_size=5,
                top_n=RERANK_TOP_N,
                llm=self.llm
            )
            query_bundle = QueryBundle(query_str=query)
            print("DEBUG: Calling LLMRerank._postprocess_nodes via asyncio.to_thread...")
            def sync_rerank_call():
                return reranker._postprocess_nodes(
                    nodes=ev.nodes,
                    query_bundle=query_bundle
                )
            reranked_nodes: List[NodeWithScore] = await asyncio.to_thread(sync_rerank_call)
            print(f"Reranking complete. Kept {len(reranked_nodes)} nodes.")
            if not reranked_nodes:
                print("Warning: Reranker removed all nodes.")
            return RerankEvent(nodes=reranked_nodes)
        except Exception as rerank_err:
            print(f"CRITICAL Rerank Error: {rerank_err}")
            traceback.print_exc()
            print("Warning: Rerank failed, proceeding with empty node list.")
            return RerankEvent(nodes=[])

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent | StartEvent:
        query = await ctx.get("original_query")
        dirname = await ctx.get("dirname")
        max_synthesis_trials = 3
        synthesis_attempt_count = 0
        workflow_trials = await ctx.get("workflow_trials", default=0)
        max_workflow_trials = 3
        if not query or not dirname:
            print("ERROR: Synthesize context missing (query/dirname).")
            return StopEvent(result="Error: Context missing.")
        if not ev.nodes:
            print("Synthesize: No nodes available for synthesis.")
            workflow_trials += 1
            await ctx.set("workflow_trials", workflow_trials)
            if workflow_trials >= max_workflow_trials:
                print(f"Max workflow retry trials ({max_workflow_trials}) reached after finding no relevant nodes for query.")
                await ctx.set("workflow_trials", 0)
                return StopEvent(result=f"Could not find relevant information for '{query}' after {max_workflow_trials} attempts.")
            else:
                print(f"No nodes for synthesis. Retrying full query process (Workflow Attempt {workflow_trials+1}/{max_workflow_trials}).")
                strategy = await ctx.get("retrieval_strategy", "hybrid")
                return StartEvent(dirname=dirname, query=query, retrieval_strategy=strategy)
        print(f"Preparing context for query '{query}' using top {len(ev.nodes)} reranked nodes.")
        augmented_nodes = []
        context_log = []
        print("Augmenting node context...")
        for node_with_score in ev.nodes:
            node = node_with_score.node
            metadata = node.metadata
            prefix_parts = []
            prefix_parts.append(f"[L{metadata.get('level', '?')}]")
            latex = metadata.get("latex_formulas", [])
            llava = metadata.get("llava_descriptions", [])
            if latex:
                prefix_parts.append(f"[Math:{len(latex)}]")
            if llava:
                prefix_parts.append(f"[Visual:{len(llava)}:{llava[0][:30]}..]")
            prefix = " ".join(prefix_parts) + " " if prefix_parts else ""
            augmented_text = f"{prefix}{node.text}".strip()
            context_log.append(f"{prefix}{node.text[:100].replace(chr(10), ' ')}...")
            temp_node = node.copy()
            temp_node.text = augmented_text
            augmented_nodes.append(NodeWithScore(node=temp_node, score=node_with_score.score))
        context_summary_for_llm = "\n".join([f"- {s}" for s in context_log[:10]])
        last_cleaned_response = "[Initial State - No Previous Response]"
        last_cleaned_critique = "[Initial State - No Previous Critique]"
        current_cleaned_response = ""
        validation_passed = False
        while synthesis_attempt_count < max_synthesis_trials:
            synthesis_attempt_count += 1
            print(f"\n--- Synthesis Attempt {synthesis_attempt_count}/{max_synthesis_trials} ---")
            raw_response_str = ""
            try:
                print("Calling initial synthesizer...")
                summarizer = CompactAndRefine(llm=self.llm, streaming=True, verbose=False)
                response_obj = await summarizer.asynthesize(query, nodes=augmented_nodes)
                accumulated_response = ""
                print("Synthesized Response (raw): ", end="", flush=True)
                if hasattr(response_obj, 'response_gen') and response_obj.response_gen is not None:
                    async for token in response_obj.response_gen:
                        print(token, end="", flush=True)
                        accumulated_response += token
                    raw_response_str = accumulated_response.strip()
                    print("\n(Streaming synthesis complete)")
                elif hasattr(response_obj, 'response'):
                    raw_response_str = str(response_obj.response).strip()
                    print(raw_response_str)
                    print("\n(Non-streaming synthesis complete)")
                else:
                    raw_response_str = str(response_obj).strip()
                    print(raw_response_str)
                    print("\n(Fallback synthesis complete)")
            except Exception as synth_regen_err:
                print(f"\nCRITICAL Synthesis/Regeneration Error (Attempt {synthesis_attempt_count}): {synth_regen_err}")
                traceback.print_exc()
                current_cleaned_response = f"[Synthesis Failed: Exception Attempt {synthesis_attempt_count} - {synth_regen_err}]"
            if not current_cleaned_response.startswith("[Synthesis Failed"):
                print(f"\nPerforming grounding check/validation (Attempt {synthesis_attempt_count})...")
                validation_prompt = (
                    f"Query: '{query}'\n\n"
                    f"Context Used (Snippets):\n{context_summary_for_llm}\n[... possibly more ...]\n\n"
                    f"Generated Answer: '{current_cleaned_response}'\n\n"
                    f"---\nCritique the Generated Answer based ONLY on the Provided Context Snippets:\n"
                    f"1. Does the answer contain claims NOT supported by the snippets?\n"
                    f"2. Does the answer directly address the Query?\n"
                    f"3. Overall assessment (briefly)?\n"
                    f"4. Final verdict (one word on the last line): 'accept' or 'reject'. Do not include reasoning in <think> blocks."
                )
                try:
                    validation_raw = await self.llm.acomplete(validation_prompt)
                    validation_llm_output = validation_raw.text.strip()
                    current_cleaned_critique = remove_think_blocks(validation_llm_output)
                    print(f"Validation LLM Output (Cleaned - Attempt {synthesis_attempt_count}):\n---\n{current_cleaned_critique}\n---")
                    if not current_cleaned_critique:
                        print("Warning: Validation LLM output is empty after cleaning. Defaulting to reject.")
                        validation_decision = "reject"
                    else:
                        verdict_line = current_cleaned_critique.split('\n')[-1].strip().lower()
                        final_verdict_match = re.search(r'\b(accept|reject)\b', verdict_line)
                        validation_decision = final_verdict_match.group(1) if final_verdict_match else "reject"
                        print(f"Validation Verdict (Attempt {synthesis_attempt_count}): '{validation_decision}'")
                        if validation_decision == "accept":
                            validation_passed = True
                except Exception as validation_err:
                    print(f"Error during validation LLM call (Attempt {synthesis_attempt_count}): {validation_err}. Defaulting to reject.")
                    current_cleaned_critique = f"[Validation Failed: Exception Attempt {synthesis_attempt_count} - {validation_err}]"
                    validation_passed = False
            else:
                print(f"Skipping validation due to synthesis failure (Attempt {synthesis_attempt_count}).")
                current_cleaned_critique = f"[Validation Skipped: Synthesis Failed Attempt {synthesis_attempt_count}]"
            if validation_passed:
                print(f"Response validated and accepted (Attempt {synthesis_attempt_count}).")
                await ctx.set("workflow_trials", 0)
                return StopEvent(result=current_cleaned_response)
            print(f"Synthesis Attempt {synthesis_attempt_count} failed validation.")
            last_cleaned_response = current_cleaned_response
            last_cleaned_critique = current_cleaned_critique
            if synthesis_attempt_count >= max_synthesis_trials:
                print(f"Maximum synthesis trials ({max_synthesis_trials}) reached without validation success.")
                break
        print(f"Failed to generate a validated response after {max_synthesis_trials} synthesis attempts.")
        await ctx.set("workflow_trials", 0)
        return StopEvent(result="the query can't be answered based on information available")