# Leaf Search System 🍃

An AI-powered leaf image search system using deep learning and vector similarity search. This system combines ResNet-50 feature extraction with FAISS vector search to enable fast and accurate botanical identification.

## 🌟 Features

### Core Search & Retrieval
- **Deep Learning Feature Extraction**: ResNet-50 (ImageNet pre-trained) for 2048-dimensional feature vectors
- **Vector Search Engine**: FAISS with cosine similarity for fast, accurate search
- **Query Preprocessing**: Automatic background removal for external images
- **Result Explanations**: Optional AI-generated explanations with confidence scores

### Image Processing
- **Background Removal**: Automatic segmentation for query images
- **Feature Normalization**: L2-normalized embeddings for cosine similarity
- **Quality Assessment**: Image validation and quality metrics

### Data Management
- **Species Management**: Add new species with automatic directory structure
- **Image Upload**: Upload and auto-index leaf images
- **Incremental Updates**: Add/update/delete entries without full rebuild
- **Data Validation**: Image format, size, and quality checks
- **Data Augmentation**: Multiple augmentation techniques for dataset expansion

### Performance & Caching
- **Feature Caching**: Cache extracted features to avoid recomputation
- **Search Caching**: Cache search results for identical queries
- **Batch Processing**: Efficient processing of multiple images
- **Async Indexing**: Non-blocking background indexing

### Infrastructure & API
- **RESTful API**: FastAPI with comprehensive OpenAPI documentation
- **Database**: SQLite for metadata storage
- **File Management**: Organized dataset structure
- **Docker Support**: Optional Milvus deployment via Docker Compose
- **Health Monitoring**: System health checks and statistics

## 📋 Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Data Management](#data-management)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                       │
│         (Web, Mobile, Python SDK, cURL, etc.)               │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP/REST
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Application                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Search   │ │ Upload   │ │ Stats    │ │  Data    │      │
│  │ Endpoint │ │ Manager  │ │ Endpoint │ │  Mgmt    │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└────┬────────────────────┬──────────────────┬───────────────┘
     │                    │                  │
     ▼                    ▼                  ▼
┌─────────┐         ┌──────────┐      ┌──────────┐
│ Feature │         │  FAISS   │      │  SQLite  │
│Extractor│         │  Index   │      │ Metadata │
│(ResNet) │         │ (Cosine) │      │ Database │
└────┬────┘         └────┬─────┘      └─────┬────┘
     │                   │                   │
     └───────────────────┴───────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
         ┌─────────┐          ┌─────────┐
         │ Dataset │          │  Cache  │
         │ Images  │          │ (Redis) │
         └─────────┘          └─────────┘
```

### Core Components

1. **Feature Extraction Layer**: 
   - ResNet-50 (PyTorch) extracts 2048-dimensional feature vectors
   - L2 normalization for cosine similarity
   - Background removal for query images

2. **Vector Search Engine**: 
   - **FAISS**: Fast, in-memory vector search with cosine similarity
   - IndexFlatIP for exact search
   - Disk-based persistence

3. **Data Processing**:
   - **Query Preprocessor**: Background removal and enhancement
   - **Data Validator**: Format, size, and quality checks
   - **Data Augmentor**: Image augmentation techniques
   - **Incremental Indexer**: Add/update/delete without full rebuild

4. **Performance Layer**:
   - **Feature Caching**: Cache extracted embeddings
   - **Search Caching**: Cache search results
   - **Batch Processing**: Efficient multi-image processing

5. **Storage & Persistence**:
   - **SQLite**: Image metadata and index mappings
   - **File System**: Organized dataset structure
   - **FAISS Index**: Persisted vector database

6. **API & Interface**:
   - **FastAPI**: RESTful endpoints with auto-documentation
   - **OpenAPI/Swagger**: Interactive API explorer

## 📦 Prerequisites

- **Python 3.10-3.13** (fully tested and compatible with Python 3.12)
- Docker and Docker Compose (optional, for Milvus deployment)
- 8GB+ RAM recommended
- 10GB+ disk space (for dataset and indices)
- GPU optional (faster feature extraction with CUDA)

> ✅ **Python 3.12 Fully Supported** - See [PYTHON_312_COMPATIBILITY.md](PYTHON_312_COMPATIBILITY.md) for details

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd search-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Milvus Services (Optional)

```bash
# Optional: Only needed if you want to use Milvus instead of FAISS
docker-compose up -d

# This will start:
# - Milvus standalone server (port 19530)
# - Attu admin interface (port 8001)
# - Supporting services (etcd, MinIO)
```

**Note**: The system works with FAISS by default. Milvus is optional.

## 🎯 Quick Start

### 1. Configure Environment

Create a `.env` file in the root directory (optional, uses defaults if not provided):

```env
# Application Settings
APP_NAME="Leaf Search System"
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Milvus Settings
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Dataset Settings
DATASET_PATH=./dataset
DATASET_METADATA=./dataset/leafsnap-dataset-images.txt
```

### 2. Ingest Dataset

```bash
# Full dataset ingestion (uses FAISS by default)
python scripts/ingest_data.py

# For testing with limited data
python scripts/ingest_data.py --limit 1000

# With custom batch size
python scripts/ingest_data.py --batch-size 64
```

**Common Options:**
- `--limit N`: Process only N images (useful for testing)
- `--batch-size N`: Batch size for feature extraction (default: 32)
- `--skip-faiss`: Skip FAISS indexing (if using Milvus only)
- `--skip-milvus`: Skip Milvus indexing (default behavior)
- `--reset-db`: Reset database before ingestion

### 3. Start API Server

```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python directly
python -m app.main
```

### 4. Access the System

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative API Docs**: http://localhost:8000/redoc (ReDoc)
- **Health Check**: http://localhost:8000/health
- **Milvus Admin (Attu)**: http://localhost:8001 (if using Milvus)

## 📊 Dataset

The system uses the **Leafsnap Dataset**, which contains:
- **185 tree species** from Northeastern North America
- **30,000+ images** including both lab and field photographs
- **Segmented images** for precise leaf analysis

### Dataset Structure

```
dataset/
├── images/
│   ├── field/          # Field photographs
│   │   └── <species>/
│   └── lab/            # Lab photographs (high-quality)
│       └── <species>/
├── segmented/
│   ├── field/          # Segmented field images
│   └── lab/            # Segmented lab images
└── leafsnap-dataset-images.txt  # Metadata file
```

### Metadata Format

The `leafsnap-dataset-images.txt` file contains:
- `file_id`: Unique identifier
- `image_path`: Path to original image
- `segmented_path`: Path to segmented image
- `species`: Scientific name of the species
- `source`: 'lab' or 'field'

## 📚 API Documentation

### Core Endpoints

#### 1. Health Check
```http
GET /health
```
Returns system health status and statistics.

**Response:**
```json
{
  "status": "healthy",
  "app_name": "Leaf Search System",
  "version": "1.0.0",
  "faiss_loaded": true,
  "database_connected": true,
  "total_images": 30866
}
```

#### 2. Search Similar Leaves
```http
POST /search
```
Upload an image to find similar leaves using FAISS with cosine similarity.

**Form Data:**
- `file` (required): Image file (JPEG, PNG)

**Query Parameters:**
- `top_k` (optional, default: 10): Number of results (1-100)
- `use_segmented` (optional, default: false): Return segmented image paths
- `explain_results` (optional, default: false): Include AI explanations and confidence scores

**Example Response:**
```json
{
  "query_image": "temp/query_1698765432_leaf.jpg",
  "results": [
    {
      "file_id": 12345,
      "image_path": "dataset/images/lab/acer_rubrum/ny1234-01-1.jpg",
      "segmented_path": "dataset/segmented/lab/acer_rubrum/ny1234-01-1.png",
      "species": "Acer rubrum",
      "source": "lab",
      "distance": 0.956
    }
  ],
  "search_time_ms": 45.6,
  "search_engine": "faiss",
  "total_results": 10,
  "similarity_metric": "cosine"
}
```

**With Explanations (explain_results=true):**
```json
{
  "results": [
    {
      "file_id": 12345,
      "species": "Acer rubrum",
      "distance": 0.956,
      "confidence_score": 87.5,
      "confidence_level": "high",
      "explanation": "Strong match based on leaf shape and vein pattern",
      "visual_similarities": ["Leaf shape matches closely", "Vein structure is similar"],
      "potential_concerns": ["Query image has slight rotation"]
    }
  ]
}
```

#### 3. Get Image Details
```http
GET /images/{file_id}
```
Get metadata for a specific image.

#### 4. List Species
```http
GET /species
```
List all species in the database.

#### 5. Get Species Images
```http
GET /species/{species_name}?limit=10
```
Get images for a specific species.

#### 6. Statistics
```http
GET /stats
```
Get database statistics.

**Response:**
```json
{
  "total_images": 30866,
  "total_species": 185,
  "total_lab_images": 23147,
  "total_field_images": 7719,
  "top_species": [
    {
      "species": "Acer rubrum",
      "total_images": 456,
      "lab_images": 340,
      "field_images": 116
    }
  ]
}
```

### Database Management Endpoints

#### 7. Add New Species
```http
POST /species/add?species_name=Acer%20rubrum&source=lab
```
Create directory structure for a new species.

#### 8. Upload Images
```http
POST /species/{species_name}/upload
```
Upload images with automatic feature extraction and indexing.

**Parameters:**
- `original_image` (file): Original leaf image (required)
- `segmented_image` (file): Segmented image (optional)
- `source` (query): 'lab' or 'field' (default: 'lab')
- `auto_index` (query): Auto-index in vector DB (default: true)
- `search_engine` (query): 'milvus' or 'faiss' (default: 'milvus')

**Example:**
```bash
curl -X POST "http://localhost:8000/species/Acer rubrum/upload" \
  -F "original_image=@leaf.jpg" \
  -F "segmented_image=@leaf_seg.png" \
  -F "source=lab" \
  -F "auto_index=true"
```

**Response:**
```json
{
  "status": "uploaded",
  "file_id": 30867,
  "species": "Acer rubrum",
  "original_path": "dataset/images/lab/acer_rubrum/ar202410-01-1.jpg",
  "segmented_path": "dataset/segmented/lab/acer_rubrum/ar202410-01-1.png",
  "indexed": true,
  "search_engine": "milvus",
  "milvus_id": 443234561234567
}
```

#### 9. Get Species Statistics
```http
GET /species/{species_name}/stats
```
Get detailed statistics for a specific species.

#### 10. Delete Image
```http
DELETE /images/{file_id}?delete_files=true
```
Delete an image from database and optionally from disk.

**Full API Documentation:**
- **Swagger UI**: http://localhost:8000/docs (interactive testing)
- **ReDoc**: http://localhost:8000/redoc (clean documentation)
- **Complete API Reference**: [API.md](API.md)

## 💡 Usage Examples

### Search for Similar Leaves

#### Python Client

```python
import requests

# Search for similar leaves
with open('test_leaf.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/search',
        files={'file': f},
        params={
            'top_k': 10,
            'explain_results': True
        }
    )

results = response.json()
for result in results['results']:
    print(f"{result['species']}: {result['distance']:.3f}")
```

#### cURL

```bash
# Basic search
curl -X POST "http://localhost:8000/search?top_k=5" \
  -F "file=@test_leaf.jpg"

# With explanations
curl -X POST "http://localhost:8000/search?top_k=10&explain_results=true" \
  -F "file=@test_leaf.jpg"

# Get statistics
curl http://localhost:8000/stats

# List all species
curl http://localhost:8000/species
```

#### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/search?top_k=10', {
  method: 'POST',
  body: formData
});

const results = await response.json();
console.log(results);
```

### Add New Species and Upload Images

#### Using Python

```python
import requests

# 1. Add a new species
response = requests.post(
    "http://localhost:8000/species/add",
    params={'species_name': 'Quercus montana', 'source': 'lab'}
)
print(response.json())

# 2. Upload an image with auto-indexing
with open('my_oak_leaf.jpg', 'rb') as f:
    response = requests.post(
        "http://localhost:8000/species/Quercus montana/upload",
        files={'original_image': f},
        params={'auto_index': True}
    )
print(response.json())

# 3. Get species statistics
response = requests.get("http://localhost:8000/species/Quercus montana/stats")
print(response.json())
```

#### Using cURL

```bash
# Add species
curl -X POST "http://localhost:8000/species/add?species_name=Acer%20newspecies"

# Upload image
curl -X POST "http://localhost:8000/species/Acer%20newspecies/upload" \
  -F "original_image=@leaf.jpg" \
  -F "segmented_image=@leaf_seg.png"

# Get stats
curl "http://localhost:8000/species/Acer%20newspecies/stats"
```

## ⚙️ Configuration

The system uses `config.py` for settings with environment variable support.

### Core Settings

```python
# Application
app_name: str = "Leaf Search System"
app_version: str = "1.0.0"
debug: bool = True
host: str = "0.0.0.0"
port: int = 8000

# Database
database_url: str = "sqlite:///./leaf_search.db"

# FAISS Index
faiss_index_path: str = "./data/faiss_index.bin"
faiss_metadata_path: str = "./data/faiss_metadata.pkl"

# Model Configuration
model_name: str = "resnet50"
feature_dim: int = 2048
batch_size: int = 32

# Search Settings
default_top_k: int = 10
max_top_k: int = 100
similarity_metric: str = "cosine"

# Dataset
dataset_path: str = "./dataset"
dataset_metadata: str = "./dataset/leafsnap-dataset-images.txt"
```

### Environment Variables

Create a `.env` file to override defaults:

```bash
# Application
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Dataset
DATASET_PATH=./dataset

# Performance
BATCH_SIZE=32
ENABLE_FEATURE_CACHING=True
ENABLE_SEARCH_CACHING=True

# Optional: Milvus (if using instead of FAISS)
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

## 🔧 Development

### Project Structure

```
search-system/
├── app/                              # Main application package
│   ├── main.py                       # FastAPI application
│   ├── models.py                     # SQLAlchemy ORM models
│   ├── schemas.py                    # Pydantic schemas
│   ├── database.py                   # Database connection
│   ├── feature_extractor.py          # ResNet-50 feature extraction
│   ├── faiss_client.py               # FAISS vector search
│   ├── milvus_client.py              # Milvus client (optional)
│   ├── upload_manager.py             # File management
│   ├── incremental_indexer.py        # Incremental updates
│   ├── data_validator.py             # Data validation
│   ├── data_augmentation.py          # Data augmentation
│   ├── cache.py                      # Caching layer
│   └── explainability.py             # Result explanations
│
├── scripts/                          # Utility scripts
│   ├── ingest_data.py                # Dataset ingestion
│   ├── test_search.py                # Search testing
│   └── test_upload.py                # Upload testing
│
├── dataset/                          # Leafsnap dataset
│   ├── images/                       # Original images
│   └── segmented/                    # Segmented images
│
├── data/                             # Generated data
│   ├── faiss_index.bin               # FAISS index file
│   └── faiss_metadata.pkl            # FAISS metadata
│
├── config.py                         # Configuration settings
├── requirements.txt                  # Python dependencies
├── docker-compose.yml                # Docker services (optional)
├── README.md                         # This file
├── API.md                            # API reference
├── FEATURES.md                       # Feature documentation
├── ARCHITECTURE.md                   # Technical architecture
└── leaf_search.db                    # SQLite database
```

### Running Tests

```bash
# Test search functionality
python scripts/test_search.py

# Test data upload
python scripts/test_upload.py
```

## 🐛 Troubleshooting

### CUDA/GPU Issues

The system automatically uses GPU if available, otherwise falls back to CPU:

```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

For CPU-only deployments:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Memory Issues

For large datasets, reduce batch size:
```bash
python scripts/ingest_data.py --batch-size 16
```

### Milvus Issues (Optional)

If using Milvus:
```bash
# Check logs
docker-compose logs milvus

# Restart
docker-compose restart milvus
```

## 📖 Additional Resources

- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Leafsnap Dataset](http://leafsnap.com/dataset/)
- [Milvus Documentation](https://milvus.io/docs) (if using Milvus)

## 📚 Documentation

- **[README.md](README.md)** - This file (overview and quickstart)
- **[API.md](API.md)** - Complete API reference
- **[FEATURES.md](FEATURES.md)** - Detailed feature documentation
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture
- **Swagger UI**: http://localhost:8000/docs (interactive API testing)
- **ReDoc**: http://localhost:8000/redoc (alternative documentation)

## 🙏 Acknowledgments

**Technologies:**
- **PyTorch** & **torchvision** - Deep learning framework
- **ResNet-50** - Feature extraction model ([He et al., 2015](https://arxiv.org/abs/1512.03385))
- **FAISS** - Efficient similarity search (Facebook AI Research)
- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - Python SQL toolkit

**Dataset:**
- **Leafsnap Dataset** - Columbia University, University of Maryland, Smithsonian Institution

---

Built for botanical research and education

**Tech Stack**: Python 3.10-3.13 • PyTorch 2.1+ • FastAPI 0.115+ • FAISS 1.7+ • SQLite

**Features**: ResNet-50 embeddings • FAISS cosine similarity • RESTful API • Background removal • Result explanations

