"""
Configuration management for Leaf Search System
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Application Settings
    app_name: str = "Leaf Search System"
    app_version: str = "1.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database Settings
    database_url: str = "sqlite:///./leaf_search.db"
    
    # Redis Settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    # Milvus Settings
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "leaf_embeddings"
    milvus_dimension: int = 2048
    
    # Faiss Settings
    faiss_index_path: str = "./data/faiss_index.bin"
    faiss_metadata_path: str = "./data/faiss_metadata.pkl"
    
    # Model Settings
    model_name: str = "resnet50"
    model_weights: str = "IMAGENET1K_V2"
    feature_dim: int = 2048
    batch_size: int = 32
    
    # Advanced Preprocessing Settings
    use_advanced_preprocessing: bool = False  # Set to True to enable
    preprocessing_profile: str = "auto"  # auto, lab, or field
    preprocessing_cache_enabled: bool = True
    
    # Query Preprocessing Settings (NEW!)
    use_query_preprocessing: bool = True  # Enable query image enhancement and normalization
    query_enhancement_enabled: bool = True  # Enable sharpness and contrast enhancement
    query_normalization_enabled: bool = True  # Enable color correction and brightness adjustment
    query_denoising_enabled: bool = True  # Enable noise reduction
    query_adaptive_preprocessing: bool = True  # Use adaptive preprocessing based on quality
    
    # Preprocessing Optimization Settings (LATEST!)
    enable_preprocessing_cache: bool = True  # Cache preprocessing results
    preprocessing_cache_ttl: int = 3600  # Preprocessing cache TTL (1 hour)
    enable_preprocessing_parallel: bool = True  # Parallel preprocessing for batches
    preprocessing_max_workers: int = 4  # Workers for parallel preprocessing
    enable_preprocessing_early_exit: bool = True  # Skip preprocessing for high-quality images
    preprocessing_quality_threshold_blur: float = 200.0  # Skip if sharper
    preprocessing_quality_threshold_contrast: float = 50.0  # Skip if higher contrast
    
    # Advanced Preprocessing Algorithm Settings (ADVANCED!)
    enable_advanced_preprocessing: bool = False  # Enable deep learning-inspired algorithms
    enable_deep_background_removal: bool = True  # Sophisticated background removal
    enable_multipoint_rotation: bool = True  # Multi-point rotation detection
    enable_leaf_aware_processing: bool = True  # Leaf-characteristic aware preprocessing
    enable_learned_parameters: bool = True  # Learn from successful matches
    learned_params_path: str = "./data/learned_params.pkl"  # Path to learned parameters
    
    # Similarity Metric Settings
    # FAISS Index is built with COSINE similarity (best for image search)
    # Index type: IndexFlatIP with normalized vectors
    similarity_metric: str = "cosine"  # Options: "cosine", "inner_product", "l2", "l1", "angular"
    
    # Dataset Settings
    dataset_path: str = "./dataset"
    dataset_metadata: str = "./dataset/leafsnap-dataset-images.txt"
    
    # Search Settings
    default_top_k: int = 10
    max_top_k: int = 100
    
    # Performance Optimization Settings (NEW!)
    enable_batch_processing: bool = True  # Parallel processing for candidates
    processing_batch_size: int = 50  # Items per batch
    max_workers: int = 4  # Maximum parallel workers
    enable_approximate_reranking: bool = True  # Use sampling for large result sets
    sampling_threshold: int = 500  # Trigger sampling above this many candidates
    sampling_ratio: float = 0.3  # Sample 30% of candidates
    min_samples: int = 100  # Minimum samples
    enable_query_optimization: bool = True  # Use JOINs for database queries
    enable_feature_caching: bool = True  # Cache feature vectors
    feature_cache_ttl: int = 3600  # Feature cache TTL (1 hour)
    enable_search_caching: bool = True  # Cache search results
    search_cache_ttl: int = 300  # Search cache TTL (5 minutes)
    
    # Storage
    upload_dir: str = "./uploads"
    temp_dir: str = "./temp"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.upload_dir,
            self.temp_dir,
            Path(self.faiss_index_path).parent,
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

