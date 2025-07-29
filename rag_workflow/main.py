import asyncio
import nest_asyncio
import traceback
from llama_index.llms.ollama import Ollama
from .workflow import RAGWorkflow
from .config import PDF_DIR

async def run_workflow():
    llm = Ollama(model="deepseek-r1:14b", request_timeout=300.0)
    workflow = RAGWorkflow(timeout=900, verbose=True, llm=llm)
    dirname = PDF_DIR
    os.makedirs(dirname, exist_ok=True)
    try:
        print("Checking Ollama connection...")
        await llm.acomplete("Respond with single word: OK")
        print("Ollama connection OK.")
    except Exception as ollama_err:
        print(f"CRITICAL Ollama Connection Error: {ollama_err}")
        traceback.print_exc()
        print("Please ensure Ollama server is running and accessible.")
        return
    while True:
        try:
            query = input("\nPlease enter query (or 'exit'): ").strip()
        except EOFError:
            print("\nInput stream closed.")
            break
        if query.lower() == "exit":
            print("Exiting.")
            break
        if not query:
            print("Query cannot be empty.")
            continue
        strategy_input = input("Select retrieval strategy (hybrid, hyde, step_back) [default: hybrid]: ").strip().lower()
        retrieval_strategy = strategy_input if strategy_input in ["hyde", "step_back"] else "hybrid"
        print(f"Using strategy: {retrieval_strategy}")
        print(f"\n--- Starting Workflow --- Query: '{query}', Strategy: {retrieval_strategy} ---")
        try:
            final_event = await workflow.run(
                query=query,
                dirname=dirname,
                retrieval_strategy=retrieval_strategy
            )
            if isinstance(final_event, StopEvent):
                final_result = final_event.result
                print(f"\n===== Workflow Finished =====")
                if isinstance(final_result, str):
                    print(f"Final Result:\n{final_result}")
                elif isinstance(final_result, VectorStoreIndex):
                    print("Workflow finished after indexing only (no query processed).")
                elif final_result is None:
                    print("Workflow finished with no explicit result (may indicate an issue).")
                else:
                    print(f"Workflow finished with result: {final_result}")
                print(f"=============================\n")
            elif final_event is None:
                print("\nWorkflow finished but returned None. This might indicate an incomplete run or internal issue.\n")
            else:
                print(f"\nWorkflow finished with unexpected event type: {type(final_event)}. Result: {final_event}\n")
        except Exception as wf_run_err:
            print(f"\n--- Workflow Execution Error ---")
            print(f"An error occurred running the workflow for query '{query}': {wf_run_err}")
            traceback.print_exc()
            print("--------------------------------")

if __name__ == "__main__":
    try:
        print("Starting RAG workflow...")
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print("Applying nest_asyncio.")
            nest_asyncio.apply()
        asyncio.run(run_workflow())
        print("RAG workflow finished.")
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user.")
    except Exception as main_err:
        print(f"An unexpected critical error occurred in the main execution block: {main_err}")
        traceback.print_exc()