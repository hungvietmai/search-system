"""
Script to create a sample .env file with default settings
"""
from pathlib import Path


ENV_TEMPLATE = """# Application Settings
APP_NAME="Leaf Search System"
APP_VERSION="1.0.0"
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Database Settings
DATABASE_URL=sqlite:///./leaf_search.db

# Milvus Settings
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=leaf_embeddings
MILVUS_DIMENSION=2048

# Faiss Settings
FAISS_INDEX_PATH=./data/faiss_index.bin
FAISS_METADATA_PATH=./data/faiss_metadata.pkl

# Model Settings
MODEL_NAME=resnet50
MODEL_WEIGHTS=IMAGENET1K_V2
FEATURE_DIM=2048
BATCH_SIZE=32

# Dataset Settings
DATASET_PATH=./dataset
DATASET_METADATA=./dataset/leafsnap-dataset-images.txt

# Search Settings
DEFAULT_TOP_K=10
MAX_TOP_K=100

# Storage
UPLOAD_DIR=./uploads
TEMP_DIR=./temp
"""


def main():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    
    if env_file.exists():
        response = input(".env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled. Existing .env file preserved.")
            return
    
    with open(env_file, 'w') as f:
        f.write(ENV_TEMPLATE)
    
    print(f"✓ Created {env_file}")
    print("\nYou can now customize the settings in .env file")
    print("Default settings are suitable for local development.")


if __name__ == "__main__":
    main()

