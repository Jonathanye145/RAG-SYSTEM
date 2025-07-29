import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def move_files():
    """Moves files from the current directory to their correct locations in the RAG Workflow project structure."""
    # Map of downloaded file names to target file paths
    file_name_mapping = {
        "src_rag_workflow_init.py": "src/rag_workflow/__init__.py",
        "src_rag_workflow_config.py": "src/rag_workflow/config.py",
        "src_rag_workflow_utils.py": "src/rag_workflow/utils.py",
        "src_rag_workflow_retriever.py": "src/rag_workflow/retriever.py",
        "src_rag_workflow_workflow.py": "src/rag_workflow/workflow.py",
        "src_rag_workflow_main.py": "src/rag_workflow/main.py",
        "src_rag_workflow_api.py": "src/rag_workflow/api.py",
        "test_workflow.py": "tests/test_workflow.py",
        "api.md": "docs/api.md",
        "requirements.txt": "requirements.txt",
        "Dockerfile": "Dockerfile",
        "docker-compose.yml": "docker-compose.yml",
        "LICENSE": "LICENSE",
        "README.md": "README.md",
        "move_files.py": "move_files.py"
    }

    # Directories to create
    directories = [
        "src/rag_workflow",
        "tests",
        "docs",
        "data/arxiv_papers",
        "data/index_storage_raptor_mm",
        "data/nougat_output",
        "data/image_output"
    ]

    # Current directory
    current_dir = os.getcwd()
    logger.info(f"Current working directory: {current_dir}")

    # List all files in the current directory
    current_files = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f))]
    logger.info(f"Files found in current directory: {current_files}")

    # Create necessary directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

    # Move files
    for source_file, target_path in file_name_mapping.items():
        source_full_path = os.path.join(current_dir, source_file)
        
        # Check if source file exists
        if not os.path.exists(source_full_path):
            logger.warning(f"Source file {source_file} not found in current directory.")
            continue
        
        # Check if source file is non-empty
        file_size = os.path.getsize(source_full_path)
        if file_size == 0:
            logger.warning(f"Source file {source_file} is empty (size: 0 bytes). Skipping move.")
            continue
        
        # Log first few lines of source file for verification
        try:
            with open(source_full_path, 'r', encoding='utf-8') as f:
                first_lines = ''.join(f.readlines()[:3])[:100]  # First 3 lines or 100 chars
            logger.info(f"Source file {source_file} size: {file_size} bytes, first lines: {first_lines}")
        except Exception as e:
            logger.error(f"Error reading {source_file}: {e}. Skipping move.")
            continue

        # Check if target file already exists
        if os.path.exists(target_path):
            logger.warning(f"Target file {target_path} already exists. Skipping move for {source_file}.")
            continue

        # Move the file
        try:
            shutil.move(source_full_path, target_path)
            logger.info(f"Moved {source_file} to {target_path}")

            # Verify target file is non-empty
            target_size = os.path.getsize(target_path)
            if target_size == 0:
                logger.error(f"Target file {target_path} is empty after move (size: 0 bytes).")
            else:
                logger.info(f"Target file {target_path} size after move: {target_size} bytes")
                # Log first few lines of target file
                try:
                    with open(target_path, 'r', encoding='utf-8') as f:
                        target_first_lines = ''.join(f.readlines()[:3])[:100]
                    logger.info(f"Target file {target_path} first lines: {target_first_lines}")
                except Exception as e:
                    logger.error(f"Error reading target file {target_path}: {e}")
        except Exception as e:
            logger.error(f"Error moving {source_file} to {target_path}: {e}")

    # Check for unmoved files
    remaining_files = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f)) and f not in file_name_mapping]
    if remaining_files:
        logger.warning("The following files were not moved as they do not match the expected structure:")
        for f in remaining_files:
            logger.warning(f" - {f}")

if __name__ == "__main__":
    try:
        logger.info("Starting file moving process for RAG Workflow project...")
        move_files()
        logger.info("File moving process completed successfully.")
    except Exception as e:
        logger.error(f"Critical error during file moving: {e}")