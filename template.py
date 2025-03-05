import os
from pathlib import Path
import logging

project_name = "ragProject"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/llm.py",
    f"src/{project_name}/components/retrieval.py",
    f"src/{project_name}/components/embeddings.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/api/__init__.py",
    f"src/{project_name}/api/routes.py",
    f"src/{project_name}/api/schemas.py",
    f"src/{project_name}/frontend/templates/index.html",
    f"src/{project_name}/frontend/static/app.js",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/train_pipeline.py",
    f"src/{project_name}/pipeline/inference_pipeline.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "vector_store/faiss_store/.gitkeep",
    "vector_store/chroma_store/.gitkeep",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "models/vector_store/.gitkeep",
    "models/llm_cache/.gitkeep",
    "cloud_deployment/vertex_ai_deploy.py",
    "cloud_deployment/gcp_setup.sh",
    "notebooks/data_exploration.ipynb",
    "notebooks/model_testing.ipynb",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trails.ipynb",
    "README.md",
    ".gitignore",
]

# Creating files and folders
for file_path in list_of_files:
    filepath = Path(file_path)
    file_dir, filename = os.path.split(filepath)

    # Create directories if they don't exist
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for the file: {filename}")

    # Create empty files if they don't exist
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, 'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")