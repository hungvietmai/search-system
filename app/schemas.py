"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    """Image source type"""
    LAB = "lab"
    FIELD = "field"


class SearchEngine(str, Enum):
    """Search engine type"""
    MILVUS = "milvus"
    FAISS = "faiss"


class SimilarityMetricType(str, Enum):
    """Similarity metric type"""
    COSINE = "cosine"
    INNER_PRODUCT = "inner_product"
    L2 = "l2"
    L1 = "l1"
    ANGULAR = "angular"


class LeafImageBase(BaseModel):
    """Base schema for leaf image"""
    file_id: int
    image_path: str
    segmented_path: Optional[str] = None
    species: str
    source: SourceType


class LeafImageCreate(LeafImageBase):
    """Schema for creating leaf image record"""
    pass


class LeafImageResponse(LeafImageBase):
    """Schema for leaf image response"""
    milvus_id: Optional[int] = None
    faiss_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ExplanationComponent(BaseModel):
    """Explanation component for search result"""
    factor: str
    score: float
    weight: float
    description: str
    contribution: float


class SearchResult(BaseModel):
    """Schema for a single search result"""
    file_id: int
    image_path: str
    segmented_path: Optional[str] = None
    species: str
    source: str
    distance: float = Field(..., description="Distance/similarity score")
    confidence_score: Optional[float] = Field(None, description="Confidence score (0-1)")
    confidence_level: Optional[str] = Field(None, description="Confidence level")
    explanation: Optional[str] = Field(None, description="Why this result matches")
    explanation_components: Optional[List[ExplanationComponent]] = Field(None, description="Detailed factors")
    visual_similarities: Optional[List[str]] = Field(None, description="Visual similarities")
    potential_concerns: Optional[List[str]] = Field(None, description="Potential concerns")
    
    class Config:
        from_attributes = True


class SearchRequest(BaseModel):
    """Schema for search request"""
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    search_engine: SearchEngine = Field(SearchEngine.MILVUS, description="Search engine to use")
    use_segmented: bool = Field(False, description="Use segmented images for search")
    
    # NEW: Query preprocessing options
    use_query_preprocessing: bool = Field(True, description="Enable query image enhancement and normalization")
    
    # NEW: Similarity metric option
    similarity_metric: Optional[SimilarityMetricType] = Field(None, description="Similarity metric to use (overrides default)")
    
    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v):
        if v < 1 or v > 100:
            raise ValueError('top_k must be between 1 and 100')
        return v


class SearchResponse(BaseModel):
    """Schema for search response"""
    query_image: str
    results: List[SearchResult]
    search_time_ms: float
    search_engine: str
    total_results: int
    
    # NEW: Information about preprocessing and metric used
    query_preprocessing_applied: bool = Field(False, description="Whether query preprocessing was applied")
    similarity_metric: Optional[str] = Field(None, description="Similarity metric used")


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str
    app_name: str
    version: str
    milvus_connected: bool
    faiss_loaded: bool
    database_connected: bool
    total_images: int


class IngestionStatus(BaseModel):
    """Schema for data ingestion status"""
    total_processed: int
    total_indexed: int
    current_species: Optional[str] = None
    progress_percentage: float
    status: str


class SpeciesInfo(BaseModel):
    """Schema for species information"""
    species: str
    total_images: int
    lab_images: int
    field_images: int


class StatsResponse(BaseModel):
    """Schema for statistics response"""
    total_images: int
    total_species: int
    total_lab_images: int
    total_field_images: int
    top_species: List[SpeciesInfo]


# Feedback and Advanced Ranking Schemas

class FeedbackType(str, Enum):
    """User feedback types"""
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    CLICKED = "clicked"
    SELECTED = "selected"
    DISMISSED = "dismissed"


class FeedbackRequest(BaseModel):
    """Schema for user feedback"""
    query_image_hash: str = Field(..., description="Hash of query image")
    result_file_id: int = Field(..., description="File ID of result")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    position: int = Field(..., ge=1, description="Position in results")
    session_id: Optional[str] = Field(None, description="Session ID")
    dwell_time: Optional[float] = Field(None, description="Time spent viewing (seconds)")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in feedback")


class FeedbackResponse(BaseModel):
    """Schema for feedback response"""
    feedback_id: int
    message: str
    

class RefineQueryRequest(BaseModel):
    """Schema for query refinement request"""
    query_image_hash: str = Field(..., description="Hash of original query")
    top_k: int = Field(10, ge=1, le=100)
    search_engine: SearchEngine = Field(SearchEngine.MILVUS)


class DiversityAlgorithm(str, Enum):
    """Diversity algorithms"""
    MMR = "mmr"
    CLUSTERING = "clustering"
    FEATURE_BASED = "feature"
    HYBRID = "hybrid"


class LTRAlgorithm(str, Enum):
    """Learning-to-rank algorithms"""
    LINEAR = "linear"
    RANKNET = "ranknet"
    LAMBDAMART = "lambdamart"


class AdvancedSearchRequest(BaseModel):
    """Schema for advanced search with LTR and diversity"""
    use_ltr: bool = Field(False, description="Use learning-to-rank")
    ltr_algorithm: LTRAlgorithm = Field(LTRAlgorithm.LINEAR, description="LTR algorithm")
    use_diversity: bool = Field(False, description="Use diversity re-ranking")
    diversity_algorithm: DiversityAlgorithm = Field(DiversityAlgorithm.MMR, description="Diversity algorithm")
    diversity_lambda: float = Field(0.7, ge=0.0, le=1.0, description="MMR lambda parameter")


