# System Architecture - Leaf Search System

Technical architecture documentation for the leaf image search system.

## 📐 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  (Web Browser, Mobile App, Python Client, cURL, etc.)       │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Search   │ │ Upload   │ │ Stats    │ │ Health   │      │
│  │ Endpoint │ │ Endpoint │ │ Endpoint │ │ Check    │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└─────┬──────────────────┬──────────────────┬────────────────┘
      │                  │                  │
      │                  │                  │
┌─────▼──────┐    ┌─────▼──────┐    ┌─────▼──────────────┐
│  Feature   │    │  Vector    │    │  Metadata         │
│ Extraction │    │  Search    │    │  Database         │
│            │    │            │    │                   │
│ ResNet-50  │    │ Milvus or  │    │  SQLite           │
│ (PyTorch)  │    │  Faiss     │    │  (SQLAlchemy)     │
└────────────┘    └─────┬──────┘    └───────────────────┘
                        │
              ┌─────────┴─────────┐
              │                   │
       ┌──────▼──────┐    ┌──────▼──────┐
       │   Milvus    │    │   Faiss     │
       │ Vector DB   │    │   Index     │
       │             │    │             │
       │ (Docker)    │    │ (In-Memory/ │
       │             │    │  Disk)      │
       └─────────────┘    └─────────────┘
```

## 🔧 Component Details

### 1. API Layer (FastAPI)

**Purpose**: REST API interface for all system operations

**Components**:
- `app/main.py`: Main FastAPI application
- `app/schemas.py`: Request/response models (Pydantic)
- `app/database.py`: Database session management

**Endpoints**:
- `GET /`: Root endpoint
- `GET /health`: System health check
- `POST /search`: Image similarity search
- `GET /images/{file_id}`: Get image details
- `GET /species`: List all species
- `GET /species/{name}`: Get species images
- `GET /stats`: System statistics

**Features**:
- CORS middleware for cross-origin requests
- Automatic API documentation (Swagger/ReDoc)
- Request validation with Pydantic
- Error handling and logging

### 2. Feature Extraction (ResNet-50)

**Purpose**: Convert images to high-dimensional feature vectors

**Components**:
- `app/feature_extractor.py`: ResNet-50 wrapper

**Process**:
1. Load image (PIL)
2. Apply background removal for query images
3. Preprocess: Resize → Center Crop → Normalize
4. Forward pass through ResNet-50 (without final classification layer)
5. Extract 2048-dimensional feature vector
6. L2 normalize the vector for cosine similarity

**Architecture**:
```
Input Image (any size)
    ↓
Background Removal (query only)
    ↓
Resize & Crop (224x224x3)
    ↓
Normalize (ImageNet stats)
    ↓
ResNet-50 Layers
    ↓
Global Average Pooling
    ↓
Feature Vector (2048-d)
    ↓
L2 Normalization
    ↓
Normalized Feature (2048-d)
```

**Optimizations**:
- Pre-trained ImageNet weights
- GPU acceleration when available
- Feature caching for repeated queries

### 3. Vector Search Engine (FAISS)

**Purpose**: Fast, local similarity search with cosine similarity

**Components**:
- `app/faiss_client.py`: FAISS client wrapper

**Features**:
- In-memory index with disk persistence
- No external dependencies
- Cosine similarity via IndexFlatIP
- Exact nearest neighbor search

**Index Configuration**:
```python
# Uses IndexFlatIP for exact cosine similarity
# - Features are L2-normalized before indexing
# - Inner product on normalized vectors = cosine similarity
# - Returns actual cosine similarity scores (0-1)
```

**Benefits**:
- Simple deployment (no Docker required)
- Fast exact search
- Reliable and deterministic results
- Easy to backup and restore

**Optional: Milvus**
- Available via Docker Compose for distributed deployments
- Supports same cosine similarity search
- Useful for production scaling

### 4. Metadata Database (SQLite)

**Purpose**: Store image metadata and relationships

**Components**:
- `app/models.py`: SQLAlchemy models
- `app/database.py`: Database connection

**Schema**:

```sql
-- Leaf Images
CREATE TABLE leaf_images (
    id INTEGER PRIMARY KEY,
    file_id INTEGER UNIQUE,
    image_path VARCHAR(500),
    segmented_path VARCHAR(500),
    species VARCHAR(200),
    source VARCHAR(50),
    milvus_id INTEGER,
    faiss_id INTEGER,
    created_at DATETIME,
    updated_at DATETIME
);

-- Search History
CREATE TABLE search_history (
    id INTEGER PRIMARY KEY,
    query_image_path VARCHAR(500),
    search_engine VARCHAR(50),
    top_k INTEGER,
    search_time_ms FLOAT,
    results TEXT,
    created_at DATETIME
);
```

**Indexes**:
- `file_id` (unique)
- `species`
- `source`
- `milvus_id`
- `faiss_id`

## 🔄 Data Flow

### Ingestion Flow

```
1. Load Dataset Metadata
   ↓
2. Verify Image Paths
   ↓
3. Insert Metadata → SQLite
   ↓
4. For each batch:
   ├─→ Load Images
   ├─→ Extract Features (ResNet-50)
   ├─→ Index in Milvus
   ├─→ Index in Faiss
   └─→ Update SQLite with IDs
   ↓
5. Save Faiss Index to Disk
   ↓
6. Complete
```

### Search Flow

```
1. Upload Query Image
   ↓
2. Save to Temp Storage
   ↓
3. Extract Features (ResNet-50)
   ↓
4. Search Vector DB (Milvus or Faiss)
   ↓
5. Get Top-K Similar Vectors
   ↓
6. Query SQLite for Metadata
   ↓
7. Return Results with Metadata
   ↓
8. Cleanup Temp File
```

## 🎯 Design Decisions

### Why ResNet-50?

- Pre-trained on ImageNet (good general features)
- Widely used and proven for image retrieval
- Good balance of accuracy vs. speed
- 2048-dimensional embeddings (reasonable size)

### Why FAISS?

- No external dependencies (unlike Milvus)
- Fast exact search with cosine similarity
- Simple deployment (single-file index)
- Disk persistence built-in
- Milvus optional for scaling needs

### Why Cosine Similarity?

- Standard for image embeddings
- Normalized measure (0-1 range)
- Works well with L2-normalized features
- Better than L2 distance for high-dimensional vectors

### Why SQLite?

- Zero configuration required
- Fast for datasets up to 100K+ images
- ACID compliant
- Python built-in support
- Easy migration to PostgreSQL if needed

### Why FastAPI?

- Modern async framework
- Automatic OpenAPI documentation
- Type validation with Pydantic
- High performance
- Easy to use and deploy

## 📊 Performance Characteristics

### Feature Extraction

| Metric | CPU | GPU (CUDA) |
|--------|-----|------------|
| Single Image | 45-60ms | 5-10ms |
| Batch (32) | 800-1200ms | 150-300ms |
| Throughput | 20-25 img/s | 100-200 img/s |

### Search Performance

| Operation | Time | Notes |
|-----------|------|-------|
| FAISS Search (30K) | 2-8ms | Exact cosine similarity |
| Feature Extraction | 45-60ms | CPU, single image |
| Feature Extraction (GPU) | 5-10ms | GPU, single image |
| Total Search | 50-70ms | CPU feature + search |
| Cached Search | <5ms | Cache hit |

### Memory Usage

| Component | Memory |
|-----------|--------|
| ResNet-50 Model | ~100MB |
| FAISS Index (30K) | ~250MB |
| API Server | ~150MB |
| **Total** | ~500MB |

## 🔐 Security Considerations

### Current Implementation

- No authentication (suitable for private networks)
- File upload validation (image formats only)
- SQL injection protection (SQLAlchemy ORM)
- Path traversal protection

### Production Recommendations

1. **Add Authentication**: JWT tokens or API keys
2. **Rate Limiting**: Prevent abuse
3. **HTTPS**: TLS/SSL encryption
4. **Input Validation**: Strict file type checking
5. **Resource Limits**: File size limits, request timeouts
6. **Network Isolation**: Firewall rules, VPC

## 📈 Scalability

### Current Scale

- ✅ Up to 100K images
- ✅ Single server deployment
- ✅ Concurrent users: ~50

### Scaling Options

#### Horizontal Scaling

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  API 1   │     │  API 2   │     │  API 3   │
└────┬─────┘     └────┬─────┘     └────┬─────┘
     └────────┬────────┴────────┬────────┘
              │                 │
         ┌────▼────┐       ┌────▼────┐
         │ Milvus  │       │ Postgres│
         │ Cluster │       │ Database│
         └─────────┘       └─────────┘
```

#### Vertical Scaling

- More CPU cores for parallel processing
- More RAM for larger Faiss indices
- GPU for faster feature extraction

## 🔮 Future Enhancements

- PostgreSQL support for larger datasets
- User authentication and rate limiting
- Advanced preprocessing options
- Web UI for visualization
- Multi-model support (EfficientNet, ViT)
- Distributed deployment with Milvus

## 📚 Technical Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| API | FastAPI | 0.104+ |
| ML Framework | PyTorch | 2.1+ |
| Model | ResNet-50 | torchvision |
| Vector DB | Milvus | 2.3+ |
| Search | Faiss | 1.7+ |
| Database | SQLite | 3.x |
| ORM | SQLAlchemy | 2.0+ |
| Validation | Pydantic | 2.5+ |
| Server | Uvicorn | 0.24+ |
| Container | Docker | 20.10+ |

## 🔗 Inter-Component Communication

```
┌─────────┐  HTTP/REST   ┌─────────┐
│ Client  │◄────────────►│ FastAPI │
└─────────┘              └────┬────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
              gRPC  │    ORM  │   Python│
              ┌─────▼─┐  ┌────▼──┐  ┌───▼───┐
              │Milvus │  │SQLite │  │ Faiss │
              └───────┘  └───────┘  └───────┘
```

## 🎓 Learning Resources

- **ResNet Paper**: https://arxiv.org/abs/1512.03385
- **Milvus Docs**: https://milvus.io/docs
- **Faiss Wiki**: https://github.com/facebookresearch/faiss/wiki
- **FastAPI Tutorial**: https://fastapi.tiangolo.com/tutorial/
- **Vector Search**: https://www.pinecone.io/learn/vector-search/

---

This architecture is designed to be:
- 🎯 **Practical**: Works out of the box
- 📈 **Scalable**: Can grow with your needs
- 🔧 **Maintainable**: Clean, modular code
- 📚 **Educational**: Learn modern ML systems

