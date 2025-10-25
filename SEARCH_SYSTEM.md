# Search System Documentation

This document explains the search system components, algorithms, and processes used in the leaf image identification application after preprocessing.

## Overview

The search system uses a two-stage approach to find similar leaf images:

1. **Initial retrieval**: Vector similarity search using FAISS with cosine similarity
2. **Re-ranking**: Learning-to-Rank (LTR) model to improve result relevance

## Search Process Flow

### 1. Query Processing

- User uploads a leaf image
- Image undergoes preprocessing (background removal, rotation correction, enhancement)
- Features are extracted using ResNet-50 model
- Features are normalized for consistency with indexed images
- Content-based caching using MD5 hash of file content (not path) to avoid cache misses due to temporary file names

### 2. Initial Retrieval (Stage 1)

- Uses FAISS index with cosine similarity
- Retrieves `top_k` candidates directly (no longer multiplies by 3 to avoid double multiplication with retrieval_multiplier)
- Cosine similarity measures the angle between feature vectors (range: -1 to 1, higher is more similar)

### 3. Two-Stage Re-ranking (Stage 2)

- Uses the Two-Stage Search Engine with diversity promotion
- When `promote_diversity=True`, requires an `emb_getter` function to retrieve embeddings for diversity calculations
- Implements Maximal Marginal Relevance (MMR) for diversity promotion
- Considers multiple factors beyond just visual similarity
- Returns top-k results after re-ranking

## Similarity Metrics

The system supports multiple similarity metrics:

### Cosine Similarity

- **Range**: [-1, 1] (higher is more similar)
- **Formula**: `cos(θ) = (A·B) / (||A|| × ||B||)`
- **Use case**: Invariant to vector magnitude, focuses on direction
- **Default for**: Main search algorithm

### L2 Distance (Euclidean)

- **Range**: [0, ∞] (lower is more similar)
- **Formula**: `√Σ(Ai - Bi)²`
- **Use case**: Measures absolute distance between vectors

### L1 Distance (Manhattan)

- **Range**: [0, ∞] (lower is more similar)
- **Formula**: `Σ|Ai - Bi|`
- **Use case**: More robust to outliers than L2

### Inner Product

- **Range**: [-∞, ∞] (higher is more similar)
- **Formula**: `Σ(Ai × Bi)`
- **Use case**: Considers both direction and magnitude

## Learning-to-Rank (LTR) System

### Ranking Features

The LTR model considers multiple features for each candidate:

1. **Vector Similarity**: Raw similarity score from initial search
2. **Cosine Similarity**: Normalized angle-based similarity
3. **Retrieval Distance**: Generic distance measure (was previously labeled as euclidean_distance despite using cosine similarity)
4. **Species Frequency**: How common the species is in the database (cached globally to avoid repeated queries)
5. **Species Popularity**: Popularity score based on frequency
6. **Image Quality Score**: Estimated quality of the image
7. **Source Score**: 1.0 for lab images, 0.5 for field images
8. **Temporal Score**: Time-based relevance (default 0.5)
9. **Diversity Score**: Promotes variety in results (default 0.5)
10. **Click-Through Rate**: Historical user engagement (default 0.1)
11. **Conversion Rate**: Historical success rate (default 0.1)

### LTR Algorithms

#### Linear LTR

- **Approach**: Weighted linear combination of features
- **Formula**: `score = Σ(weights[i] × feature[i])`
- **Default weights**:
  - Vector similarity: 0.30
  - Cosine similarity: 0.20
  - Retrieval distance: -0.10 (negative because lower is better)
  - Species frequency: 0.10
- Species popularity: 0.05
- Image quality score: 0.10
- Source score: 0.05
- Temporal score: 0.05
- Diversity score: 0.05
- CTR: 0.05
- Conversion rate: 0.05

#### Pairwise LTR (RankNet-style)

- **Approach**: Learns to predict relative ordering of pairs
- **Method**: Compares pairs of results and learns which should be ranked higher
- **Loss function**: Cross-entropy based on predicted vs actual relative ordering

### Two-Stage Search Engine

#### Stage 1: Retrieval

- Retrieves more candidates than needed (`top_k * retrieval_multiplier`)
- Default multiplier: 50 (can retrieve up to 1000 candidates)
- Uses vector similarity to get initial candidates
- The retrieval calculation happens in the `stage1_retrieval` method at line 99: `retrieve_k = min(top_k * self.retrieval_multiplier, 1000)`.

#### Stage 2: Re-ranking

- Computes multiple scores for each candidate:
  - Vector similarity score (from initial search)
  - Species frequency score (common species preferred, cached globally to avoid repeated database queries)
  - Source preference score (lab images preferred)
  - Diversity score (penalizes duplicate species using MMR approach)
- Combines scores using weighted formula:
  ```
  final_score = w1×vector_score + w2×frequency_score + w3×source_score + w4×diversity_score
  ```
- Default weights:
  - Vector similarity: 0.70
- Species prior: 0.05
- Source preference: 0.15
- Diversity: 0.0 (MMR handles diversity separately)

## Search Endpoints

### Basic Search (`/search`)

- Accepts an image file upload
- Returns top-k similar images
- Optional features:
  - AI explanations for matches
- Learning-to-Rank re-ranking
- Segmented image paths
- Dataset-only optimized search with two-stage engine
- Implements global LTR model caching to avoid reloading for every request
- Uses emb_getter function when promote_diversity=True in two-stage search

### Optimized Search (`/search-optimized`)

- Enhanced version with performance optimizations
- Uses approximate re-ranking for large result sets
- Implements caching for better performance
- Optimized database queries with global species caching
- Content-based caching instead of path-based caching

## Performance Optimizations

### Caching

- Feature caching: Stores extracted features to avoid recomputation
- Search result caching: Caches identical queries
- Preprocessing caching: Stores preprocessed images
- Global LTR model caching: Avoids reloading models for each request
- Global species frequency caching: Prevents repeated database queries in loops

### Approximate Re-ranking

- For large result sets (>sampling_threshold), uses sampling
- Reduces computation time while maintaining quality
- Configurable sampling ratio

### Batch Processing

- Processes multiple candidates simultaneously
- Optimized database queries
- Parallel feature extraction

## Relevance Feedback System

The system collects user feedback to improve search quality:

- Records user interactions (clicks, selections, dismissals)
- Updates ranking models based on feedback
- Implements Rocchio algorithm for query refinement
- Tracks click-through rates and conversion rates

## Configuration Parameters

### Search Settings

- `top_k`: Number of results to return (default: 10)
- `retrieval_multiplier`: How many candidates to retrieve initially (default: 50, capped at 100)
- `use_ltr`: Whether to apply Learning-to-Rank (default: True)
- `promote_diversity`: Whether to promote diversity in results (default: True)

### Performance Settings

- `enable_batch_processing`: Batch processing for large sets
- `batch_size`: Size of processing batches
- `enable_approximate_reranking`: Use sampling for large sets
- `sampling_threshold`: When to switch to approximate re-ranking
- `sampling_ratio`: Proportion of candidates to sample

### LTR Settings

- `learning_rate`: Rate of model updates
- `iterations`: Number of training iterations
- `algorithm`: LTR algorithm to use (linear, ranknet, lambdamart)

## Recent Improvements and Fixes

### Fixed Issues

1. **Missing emb_getter**: The two-stage search engine now properly passes an emb_getter function when promote_diversity=True to avoid crashes or silent fallbacks.

2. **Top-k multiplication**: Fixed the double multiplication issue where top_k was multiplied twice (once when passed to the engine and again internally by retrieval_multiplier).

3. **Cache key consistency**: Updated cache key generation to use MD5 hash of file content instead of temporary file paths, ensuring proper cache hits.

4. **LTR model caching**: Moved LTR model cache to module scope to avoid reloading models for each request.

5. **Species frequency caching**: Implemented global species cache to avoid rebuilding it in each loop iteration.

6. **Distance naming consistency**: Renamed euclidean_distance to retrieval_distance to reflect that FAISS uses cosine similarity, not euclidean distance.

### Expected Improvements

- **Initial search**: FAISS with cosine similarity provides fast, accurate results
- **Re-ranking**: Learning-to-Rank adds +15-20% relevance improvement
- **Two-stage approach**: Better balance of speed and accuracy with diversity promotion
- **Relevance feedback**: Continuous improvement from user interactions
- **Performance optimizations**: Significant speed improvements for large datasets
- **Cache efficiency**: Better hit rates due to content-based caching
- **Memory efficiency**: Reduced database queries through global caching
