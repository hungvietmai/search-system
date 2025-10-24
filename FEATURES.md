# Leaf Search System - Features Documentation

Complete guide to features in the Leaf Search System.

## Table of Contents

1. [Core Search Features](#core-search-features)
2. [Image Processing](#image-processing)
3. [Data Management](#data-management)
4. [Performance Features](#performance-features)
5. [API Features](#api-features)

---

## Core Search Features

### Vector Similarity Search

Fast, accurate leaf image search using deep learning embeddings.

**How It Works:**
1. Query image is processed through ResNet-50
2. Extracts 2048-dimensional feature vector
3. L2-normalized for cosine similarity
4. FAISS searches for nearest neighbors
5. Returns most similar leaf images

**Key Features:**
- **Cosine Similarity**: Industry-standard metric for image search
- **Exact Search**: No approximation, always finds true nearest neighbors
- **Fast Retrieval**: 2-8ms search time for 30K images
- **Configurable top-k**: Return 1-100 results

**Usage:**
```bash
# Basic search
POST /search?top_k=10
file: image.jpg

# With segmented images
POST /search?top_k=10&use_segmented=true
file: image.jpg
```

### Result Explanations

Optional AI-generated explanations for search results.

**Features:**
- Confidence scoring (0-100)
- Confidence levels (high/medium/low)
- Visual similarity descriptions
- Potential concerns or warnings

**Example Response:**
```json
{
  "confidence_score": 87.5,
  "confidence_level": "high",
  "explanation": "Strong match based on leaf shape and vein pattern",
  "visual_similarities": [
    "Leaf shape matches closely",
    "Vein structure is similar"
  ],
  "potential_concerns": [
    "Query image has slight rotation"
  ]
}
```

**Usage:**
```bash
POST /search?explain_results=true
```

---

## Image Processing

### Background Removal

Automatic background segmentation for query images.

**Features:**
- Removes backgrounds from uploaded images
- Focuses on leaf features
- Improves search accuracy for external images
- Applied automatically to query images

**Process:**
1. Detect leaf vs. background
2. Create binary mask
3. Extract leaf region
4. Use for feature extraction

### Feature Extraction

ResNet-50 based deep learning feature extraction.

**Architecture:**
- Pre-trained on ImageNet dataset
- 2048-dimensional embeddings
- L2-normalized vectors
- GPU acceleration when available

**Performance:**
- CPU: 45-60ms per image
- GPU: 5-10ms per image
- Batch processing supported

---

## Data Management

### Species Management

Add and manage tree species in the database.

**Endpoints:**
```http
# Add new species
POST /species/add?species_name=Acer rubrum

# Get species statistics
GET /species/{species_name}/stats

# List all species
GET /species
```

**Features:**
- Automatic directory creation
- Lab and field image organization
- Statistics tracking

### Image Upload

Upload leaf images with automatic indexing.

**Endpoints:**
```http
POST /species/{species_name}/upload
```

**Form Data:**
- `original_image`: Original leaf image (required)
- `segmented_image`: Segmented image (optional)

**Query Parameters:**
- `source`: 'lab' or 'field' (default: 'lab')
- `auto_index`: Auto-index in FAISS (default: true)

**Features:**
- Automatic feature extraction
- Immediate FAISS indexing
- Database metadata storage
- File organization

**Example:**
```python
import requests

with open('oak_leaf.jpg', 'rb') as f:
    response = requests.post(
        "http://localhost:8000/species/Quercus alba/upload",
        files={'original_image': f},
        params={'auto_index': True}
    )
```

### Incremental Indexing

Update the search index without full rebuild.

**Endpoints:**
```http
# Add single entry
POST /data/incremental-index

# Update entry
PUT /data/update-index/{file_id}

# Delete entry
DELETE /data/index/{file_id}

# Synchronize index
POST /data/sync-index
```

**Benefits:**
- Much faster than full rebuild
- No downtime during updates
- Audit trail of changes

### Data Validation

Validate image quality before indexing.

**Endpoints:**
```http
# Validate single image
POST /data/validate
file: image.jpg

# Batch validation
POST /data/validate-batch
{
  "image_paths": ["img1.jpg", "img2.jpg"]
}
```

**Checks:**
- Image format (JPEG, PNG)
- File size (1KB - 10MB)
- Resolution requirements
- Quality metrics (blur, brightness, contrast)
- Corruption detection

**Example Response:**
```json
{
  "valid": true,
  "score": 85.5,
  "issues": [
    {
      "rule": "quality",
      "severity": "warning",
      "message": "Image appears slightly blurry"
    }
  ],
  "metrics": {
    "blur_score": 25.3,
    "brightness_score": 65.2,
    "contrast_score": 42.1
  }
}
```

### Data Augmentation

Generate augmented versions of images.

**Endpoints:**
```http
# Augment single image
POST /data/augment
file: leaf.jpg
augmentations_count: 5
profile: standard

# Augment dataset
POST /data/augment-dataset
{
  "image_paths": ["img1.jpg", "img2.jpg"],
  "output_dir": "augmented",
  "augmentations_per_image": 5,
  "profile": "standard"
}

# Stratified augmentation (balance species)
POST /data/stratified-augmentation
```

**Augmentation Techniques:**
- Geometric: Rotation, Flip, Scale, Crop
- Color: Brightness, Contrast, Saturation, Hue
- Effects: Gaussian Noise, Blur, Sharpen

**Profiles:**
- **Minimal**: 3 basic transformations
- **Standard**: 8 balanced augmentations  
- **Aggressive**: 15+ techniques

---

## Performance Features

### Caching

Multi-layer caching for fast repeated queries.

**Cache Types:**
- **Feature Cache**: Extracted embeddings (1 hour TTL)
- **Search Cache**: Query results (5 minutes TTL)

**Endpoints:**
```http
# View cache statistics
GET /cache/stats

# Clear caches
POST /cache/clear
```

**Performance Impact:**
- Cache hit: <5ms response time
- Cache miss: 50-70ms (feature extraction + search)
- Typical hit rate: 40-60%

**Example Stats:**
```json
{
  "feature_cache": {
    "size": 856,
    "capacity": 1000,
    "hit_rate": 0.62
  },
  "search_cache": {
    "size": 432,
    "capacity": 1000,
    "hit_rate": 0.45
  }
}
```

### Asynchronous Indexing

Background indexing for non-blocking operations.

**Endpoints:**
```http
# Queue single image
POST /index/async
{
  "image_path": "dataset/images/lab/maple/leaf.jpg",
  "file_id": 12345
}

# Batch indexing
POST /index/async/batch
{
  "file_ids": [1, 2, 3],
  "image_paths": ["img1.jpg", "img2.jpg", "img3.jpg"]
}

# Check status
GET /index/status/{task_id}

# List tasks
GET /index/tasks

# Get statistics
GET /index/stats
```

**Benefits:**
- Non-blocking API
- Progress tracking
- Automatic retry on failures

---

## API Features

### RESTful API

FastAPI-based REST API with auto-documentation.

**Features:**
- 25+ endpoints
- OpenAPI/Swagger UI
- ReDoc documentation
- Request validation
- Type safety with Pydantic

**Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

### Health Monitoring

System health checks and statistics.

**Endpoints:**
```http
# Health check
GET /health

# System statistics
GET /stats

# Species statistics  
GET /species/{name}/stats

# Optimization stats
GET /optimization/stats
```

**Health Response:**
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

**Statistics Response:**
```json
{
  "total_images": 30866,
  "total_species": 185,
  "total_lab_images": 23147,
  "total_field_images": 7719,
  "top_species": [...]
}
```

---

## Feature Comparison

| Feature | Status | Performance | Use Case |
|---------|--------|------------|----------|
| Vector Search | ✅ Implemented | 2-8ms | Core functionality |
| Background Removal | ✅ Implemented | 50-100ms | Query preprocessing |
| Result Explanations | ✅ Implemented | +5-10ms | Interpretability |
| Feature Caching | ✅ Implemented | -95% (cached) | Repeated queries |
| Image Upload | ✅ Implemented | 50-100ms | Data management |
| Data Validation | ✅ Implemented | 50-100ms | Quality control |
| Data Augmentation | ✅ Implemented | 100-200ms/img | Dataset expansion |
| Async Indexing | ✅ Implemented | Non-blocking | Background processing |

---

## Configuration

Key configuration options in `config.py`:

```python
# Search
similarity_metric: str = "cosine"
default_top_k: int = 10
max_top_k: int = 100

# Performance  
enable_feature_caching: bool = True
enable_search_caching: bool = True
batch_size: int = 32

# Dataset
dataset_path: str = "./dataset"
faiss_index_path: str = "./data/faiss_index.bin"
```

---

For more details, see:
- **README.md**: Setup and quickstart
- **API.md**: Complete API reference
- **ARCHITECTURE.md**: Technical architecture
