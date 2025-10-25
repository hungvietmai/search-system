# Leaf Search System - API Documentation

Complete REST API reference for the Leaf Search System.

## Base URL

```
http://localhost:8000
```

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs - Interactive API testing
- **ReDoc**: http://localhost:8000/redoc - Clean documentation format

## API Overview

**Total Endpoints**: 25+  
**Authentication**: None (development/internal use)  
**Response Format**: JSON  
**File Upload**: Multipart form-data

## Table of Contents

1. [Core Search](#core-search)
2. [Species & Images](#species--images)
3. [Data Management](#data-management)
4. [Performance & Caching](#performance--caching)
5. [System Health](#system-health)

---

## Core Search

### Search Similar Leaves

Find similar leaf images using FAISS with cosine similarity.

```http
POST /search
```

**Form Data**:

- `file` (required): Image file (JPEG, PNG, max 10MB)

**Query Parameters**:

- `top_k` (optional, default: 10): Number of results (1-100)
- `use_segmented` (optional, default: false): Return segmented image paths
- `explain_results` (optional, default: false): Include AI explanations

**Example Request**:

```bash
curl -X POST "http://localhost:8000/search?top_k=10&explain_results=true" \
  -F "file=@test_leaf.jpg"
```

**Example Response (Basic)**:

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

**Example Response (with explanations)**:

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
      "visual_similarities": [
        "Leaf shape matches closely",
        "Vein structure is similar"
      ],
      "potential_concerns": ["Query image has slight rotation"]
    }
  ]
}
```

---

### Improved Search with Learning-to-Rank

Find similar leaf images using FAISS for initial search, then re-rank results using a trained Learning-to-Rank model for better relevance. This endpoint considers multiple factors beyond just visual similarity, including species frequency, source quality, and user interaction patterns.

```http
POST /search-improved
```

**Form Data**:

- `file` (required): Image file (JPEG, PNG, max 10MB)

**Query Parameters**:

- `top_k` (optional, default: 10): Number of results (1-100)
- `use_segmented` (optional, default: false): Return segmented image paths
- `explain_results` (optional, default: false): Include AI explanations

**Example Request**:

```bash
curl -X POST "http://localhost:8000/search-improved?top_k=10&explain_results=true" \
  -F "file=@test_leaf.jpg"
```

**Example Response**:

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
  "search_time_ms": 65.2, // May be slightly slower due to re-ranking
  "search_engine": "faiss_with_ltr",
  "total_results": 10,
  "similarity_metric": "cosine_with_ltr_reranking"
}
```

**Key Differences from Basic Search**:

- Uses the trained_ltr_model.pkl to re-rank results based on multiple features
- Considers species frequency, image source quality, and other factors
- May return different ordering than basic similarity search
- Better relevance for complex queries where visual similarity alone is insufficient
- Falls back to basic similarity search if the trained model is unavailable

---

## Species & Images

### List Species

Get all species in the database.

```http
GET /species
```

**Response**:

```json
["Acer rubrum", "Quercus alba", "Fagus grandifolia"]
```

### Get Species Images

Get images for a specific species.

```http
GET /species/{species_name}
```

**Query Parameters**:

- `limit` (optional, default: 10): Max images to return (1-100)

**Example**:

```bash
curl "http://localhost:8000/species/Acer%20rubrum?limit=5"
```

**Response**:

```json
[
  {
    "file_id": 12345,
    "image_path": "dataset/images/lab/acer_rubrum/ny1234-01-1.jpg",
    "segmented_path": "dataset/segmented/lab/acer_rubrum/ny1234-01-1.png",
    "species": "Acer rubrum",
    "source": "lab"
  }
]
```

### Get Image Details

Get metadata for a specific image.

```http
GET /images/{file_id}
```

**Response**:

```json
{
  "file_id": 12345,
  "image_path": "dataset/images/lab/acer_rubrum/ny1234-01-1.jpg",
  "segmented_path": "dataset/segmented/lab/acer_rubrum/ny1234-01-1.png",
  "species": "Acer rubrum",
  "source": "lab",
  "faiss_id": 12344,
  "created_at": "2024-10-23T10:30:00"
}
```

### Add New Species

Create a new species entry.

```http
POST /species/add
```

**Query Parameters**:

- `species_name` (required): Scientific name (e.g., "Acer rubrum")
- `source` (optional, default: "lab"): "lab" or "field"

**Example**:

```bash
curl -X POST "http://localhost:8000/species/add?species_name=Quercus%20montana&source=lab"
```

**Response**:

```json
{
  "status": "created",
  "species": "Quercus montana",
  "folder_name": "quercus_montana",
  "images_directory": "dataset/images/lab/quercus_montana",
  "segmented_directory": "dataset/segmented/lab/quercus_montana"
}
```

### Upload Images

Upload images for a species with automatic indexing.

```http
POST /species/{species_name}/upload
```

**Form Data**:

- `original_image` (required): Original leaf image
- `segmented_image` (optional): Segmented image

**Query Parameters**:

- `source` (optional, default: "lab"): "lab" or "field"
- `auto_index` (optional, default: true): Auto-index in FAISS

**Example**:

```bash
curl -X POST "http://localhost:8000/species/Acer%20rubrum/upload" \
  -F "original_image=@leaf.jpg" \
  -F "segmented_image=@leaf_seg.png" \
  -F "source=lab" \
  -F "auto_index=true"
```

**Response**:

```json
{
  "status": "uploaded",
  "file_id": 30867,
  "species": "Acer rubrum",
  "original_path": "dataset/images/lab/acer_rubrum/ar202410-01-1.jpg",
  "segmented_path": "dataset/segmented/lab/acer_rubrum/ar202410-01-1.png",
  "indexed": true,
  "faiss_id": 30866
}
```

### Get Species Statistics

Get detailed statistics for a species.

```http
GET /species/{species_name}/stats
```

**Response**:

```json
{
  "species": "Acer rubrum",
  "folder_name": "acer_rubrum",
  "total_images": 456,
  "lab_images": 340,
  "field_images": 116,
  "segmented_images": 456,
  "exists": true
}
```

### Delete Image

Delete an image from the database.

```http
DELETE /images/{file_id}
```

**Query Parameters**:

- `delete_files` (optional, default: false): Also delete physical files

**Response**:

```json
{
  "file_id": 12345,
  "species": "Acer rubrum",
  "deleted_from_db": true,
  "deleted_files": true
}
```

---

## Data Management

### Incremental Index - Add

Add a single image to the index.

```http
POST /data/incremental-index
```

**Request Body**:

```json
{
  "file_id": 12345,
  "image_path": "dataset/images/lab/maple/img_001.jpg",
  "species": "Acer rubrum",
  "source": "lab",
  "segmented_path": "dataset/segmented/lab/maple/img_001.png"
}
```

**Response**:

```json
{
  "success": true,
  "operation": "add",
  "file_id": 12345,
  "timestamp": "2024-10-23T10:30:00"
}
```

### Incremental Index - Update

Update an existing index entry.

```http
PUT /data/update-index/{file_id}
```

**Query Parameters**:

- `new_image_path` (optional): Update image path
- `new_species` (optional): Update species name

**Response**:

```json
{
  "success": true,
  "operation": "update",
  "file_id": 12345,
  "timestamp": "2024-10-23T10:30:00"
}
```

### Incremental Index - Delete

Delete an entry from the index.

```http
DELETE /data/index/{file_id}
```

**Response**:

```json
{
  "success": true,
  "operation": "delete",
  "file_id": 12345,
  "timestamp": "2024-10-23T10:30:00"
}
```

### Synchronize Index

Synchronize index with database.

```http
POST /data/sync-index
```

**Response**:

```json
{
  "message": "Index synchronization complete",
  "statistics": {
    "total_synced": 30866,
    "added": 5,
    "removed": 2,
    "updated": 3
  }
}
```

### Get Index Changes

Get recent index changes.

```http
GET /data/index-changes
```

**Query Parameters**:

- `operation` (optional): Filter by operation ("add", "update", "delete", "sync")
- `limit` (optional, default: 100): Max changes to return (1-1000)

**Response**:

```json
{
  "total_changes": 150,
  "changes": [
    {
      "operation": "add",
      "file_id": 12345,
      "species": "Acer rubrum",
      "timestamp": "2024-10-23T10:30:00",
      "success": true
    }
  ]
}
```

### Validate Image

Validate an image before indexing.

```http
POST /data/validate
```

**Form Data**:

- `file` (required): Image to validate

**Query Parameters**:

- `species` (optional): Expected species (for content check)

**Response**:

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
    "contrast_score": 42.1,
    "content_score": 75.0
  }
}
```

### Batch Validate

Validate multiple images.

```http
POST /data/validate-batch
```

**Request Body**:

```json
{
  "image_paths": ["img1.jpg", "img2.jpg", "img3.jpg"]
}
```

**Response**:

```json
{
  "statistics": {
    "total_valid": 2,
    "total_invalid": 1,
    "average_score": 75.3
  },
  "results": {
    "img1.jpg": {
      "valid": true,
      "score": 85.5
    },
    "img2.jpg": {
      "valid": true,
      "score": 90.2
    },
    "img3.jpg": {
      "valid": false,
      "score": 45.1
    }
  }
}
```

### Augment Image

Generate augmented versions of an image.

```http
POST /data/augment
```

**Form Data**:

- `file` (required): Image to augment

**Query Parameters**:

- `augmentations_count` (optional, default: 5): Number of augmentations (1-20)
- `profile` (optional, default: "standard"): "minimal", "standard", or "aggressive"

**Response**:

```json
{
  "original_filename": "leaf.jpg",
  "augmentations_generated": 5,
  "profile": "standard",
  "augmentations": [
    {
      "version": 1,
      "augmentations_applied": ["rotation", "flip"]
    }
  ]
}
```

### Augment Dataset

Augment multiple images.

```http
POST /data/augment-dataset
```

**Request Body**:

```json
{
  "image_paths": ["img1.jpg", "img2.jpg"],
  "output_dir": "augmented",
  "augmentations_per_image": 5,
  "profile": "standard"
}
```

**Response**:

```json
{
  "message": "Dataset augmentation complete",
  "original_count": 2,
  "augmented_count": 10,
  "output_directory": "augmented",
  "profile": "standard"
}
```

### Stratified Augmentation

Balance species distribution through augmentation.

```http
POST /data/stratified-augmentation
```

**Query Parameters**:

- `output_dir` (required): Output directory
- `target_count` (optional, default: 100): Target images per species (10-1000)
- `profile` (optional, default: "standard"): Augmentation profile

**Response**:

```json
{
  "message": "Stratified augmentation complete",
  "target_count": 100,
  "species_count": 185,
  "output_directory": "balanced_dataset",
  "profile": "standard"
}
```

---

## Performance & Caching

### Get Cache Statistics

View caching performance metrics.

```http
GET /cache/stats
```

**Response**:

```json
{
  "feature_cache": {
    "size": 856,
    "capacity": 1000,
    "hit_rate": 0.62,
    "hits": 15234,
    "misses": 9123
  },
  "search_cache": {
    "size": 432,
    "capacity": 1000,
    "hit_rate": 0.45,
    "hits": 8932,
    "misses": 11023
  }
}
```

### Clear Caches

Clear all or specific caches.

```http
POST /cache/clear
```

**Query Parameters**:

- `cache_type` (optional): "feature", "search", or "all" (default: "all")

**Response**:

```json
{
  "status": "cleared",
  "cache_type": "all",
  "items_cleared": 1288
}
```

### Submit Async Indexing

Queue image for background indexing.

```http
POST /index/async
```

**Query Parameters**:

- `image_path` (required): Path to image
- `file_id` (required): File ID

**Response**:

```json
{
  "task_id": "task_abc123",
  "message": "Indexing task submitted",
  "status_url": "/index/status/task_abc123"
}
```

### Batch Async Indexing

Queue multiple images for indexing.

```http
POST /index/async/batch
```

**Request Body**:

```json
{
  "file_ids": [1, 2, 3],
  "image_paths": ["img1.jpg", "img2.jpg", "img3.jpg"]
}
```

**Response**:

```json
{
  "task_id": "batch_xyz789",
  "message": "Batch indexing task submitted (3 images)"
}
```

### Get Indexing Status

Check status of an indexing task.

```http
GET /index/status/{task_id}
```

**Response**:

```json
{
  "task_id": "task_abc123",
  "status": "completed",
  "progress": 100,
  "result": {
    "file_id": 12345,
    "faiss_id": 12344,
    "indexed": true
  },
  "created_at": "2024-10-23T10:30:00",
  "completed_at": "2024-10-23T10:30:15"
}
```

### List Indexing Tasks

List all indexing tasks.

```http
GET /index/tasks
```

**Response**:

```json
{
  "tasks": [
    {
      "task_id": "task_abc123",
      "status": "completed"
    }
  ],
  "total": 1
}
```

### Get Indexing Statistics

Get indexer performance statistics.

```http
GET /index/stats
```

**Response**:

```json
{
  "pending_tasks": 5,
  "processing_tasks": 2,
  "completed_tasks": 143,
  "failed_tasks": 3
}
```

### Get Optimization Statistics

Get performance optimization statistics.

```http
GET /optimization/stats
```

**Response**:

```json
{
  "caching": {
    "feature_cache": {
      "hit_rate": 0.62
    },
    "search_cache": {
      "hit_rate": 0.45
    }
  },
  "config": {
    "enable_batch_processing": true,
    "batch_size": 32,
    "enable_feature_caching": true,
    "enable_search_caching": true
  }
}
```

---

## System Health

### Health Check

Get system health status.

```http
GET /health
```

**Response**:

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

### Statistics

Get system-wide statistics.

```http
GET /stats
```

**Response**:

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

---

## Error Responses

All endpoints return standardized error responses:

### Validation Error (422)

```json
{
  "detail": [
    {
      "loc": ["query", "top_k"],
      "msg": "ensure this value is less than or equal to 100",
      "type": "value_error.number.not_le"
    }
  ]
}
```

### Not Found (404)

```json
{
  "detail": "Image with file_id 12345 not found"
}
```

### Server Error (500)

```json
{
  "detail": "Internal server error: Feature extraction failed"
}
```

## HTTP Status Codes

- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `413 Payload Too Large`: File too large (>10MB)
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

---

## Rate Limiting

Currently no rate limiting is enforced. For production deployments, consider implementing:

- 100 requests/minute for search endpoints
- 20 requests/minute for upload endpoints
- 300 requests/minute for other endpoints

---

## API Versioning

**Current Version**: v1.0.0

The API currently does not use URL versioning. Future versions may use `/v1/` prefixes.

---

## Client Examples

### Python

```python
import requests

# Search
with open('leaf.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/search',
        files={'file': f},
        params={'top_k': 10, 'explain_results': True}
    )
results = response.json()

# Upload
with open('new_leaf.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/species/Acer rubrum/upload',
        files={'original_image': f},
        params={'auto_index': True}
    )
```

### JavaScript

```javascript
// Search
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const response = await fetch(
  "http://localhost:8000/search?top_k=10&explain_results=true",
  {
    method: "POST",
    body: formData,
  }
);
const results = await response.json();
```

### cURL

```bash
# Search
curl -X POST "http://localhost:8000/search?top_k=10" \
  -F "file=@leaf.jpg"

# Upload
curl -X POST "http://localhost:8000/species/Acer%20rubrum/upload" \
  -F "original_image=@leaf.jpg" \
  -F "auto_index=true"

# Statistics
curl "http://localhost:8000/stats"
```

---

## Support & Resources

**Documentation**:

- [README.md](README.md) - Setup and quickstart
- [FEATURES.md](FEATURES.md) - Detailed features
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture

**API**:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI Spec: http://localhost:8000/openapi.json

**System Status**:

- Health: http://localhost:8000/health
- Stats: http://localhost:8000/stats
- Cache Stats: http://localhost:8000/cache/stats

---

**Built with FastAPI** • **25+ Endpoints** • **v1.0.0**
