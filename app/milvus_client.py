"""
Milvus vector database client for similarity search
Now with multi-metric support!
"""
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from config import settings
from app.similarity_metrics import SimilarityMetric, convert_metric_to_milvus

logger = logging.getLogger(__name__)


class MilvusClient:
    """Client for interacting with Milvus vector database with multi-metric support"""
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, 
                 collection_name: Optional[str] = None, metric: Optional[SimilarityMetric] = None):
        """
        Initialize Milvus client
        
        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of the collection to use
            metric: Similarity metric to use (default: L2)
        """
        self.host = host or settings.milvus_host
        self.port = port or settings.milvus_port
        self.collection_name = collection_name or settings.milvus_collection_name
        
        # Convert string metric to enum if needed
        if metric:
            self.metric = metric
        else:
            metric_str = getattr(settings, 'similarity_metric', 'l2')
            self.metric = SimilarityMetric(metric_str) if isinstance(metric_str, str) else metric_str
        
        self.collection = None
        self.connected = False
        
        self._connect()
    
    def _connect(self):
        """Connect to Milvus server"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            self.connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self.connected = False
            raise
    
    def create_collection(self, dimension: Optional[int] = None, drop_existing: bool = False):
        """
        Create a new collection for storing embeddings
        
        Args:
            dimension: Dimension of the embedding vectors
            drop_existing: Whether to drop existing collection
        """
        if dimension is None:
            dimension = settings.milvus_dimension
        
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                if drop_existing:
                    logger.info(f"Dropping existing collection: {self.collection_name}")
                    utility.drop_collection(self.collection_name)
                else:
                    logger.info(f"Collection {self.collection_name} already exists")
                    self.collection = Collection(self.collection_name)
                    return
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="file_id", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Leaf embeddings collection"
            )
            
            # Create collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            # Create index for efficient similarity search
            milvus_metric = convert_metric_to_milvus(self.metric)
            index_params = {
                "metric_type": milvus_metric,
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info(f"Collection {self.collection_name} created with dimension {dimension}, metric {milvus_metric}")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def get_collection_metric(self) -> Optional[str]:
        """
        Get the metric type used by the existing collection's index
        
        Returns:
            Milvus metric type string (e.g., "L2", "IP", "COSINE") or None
        """
        try:
            if self.collection is None:
                self.collection = Collection(self.collection_name)
            
            # Get index info
            indexes = self.collection.indexes
            if indexes and len(indexes) > 0:
                index = indexes[0]
                params = index.params
                metric_type = params.get('metric_type', None)
                logger.info(f"Collection {self.collection_name} uses metric: {metric_type}")
                return metric_type
            else:
                logger.warning(f"No index found for collection {self.collection_name}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to get collection metric: {e}")
            return None
    
    def load_collection(self):
        """Load collection into memory for search"""
        try:
            if self.collection is None:
                self.collection = Collection(self.collection_name)
            
            # Try to detect the actual metric used by the collection
            actual_metric = self.get_collection_metric()
            if actual_metric:
                expected_metric = convert_metric_to_milvus(self.metric)
                if actual_metric != expected_metric:
                    logger.warning(
                        f"Collection metric mismatch! Collection uses '{actual_metric}' "
                        f"but client is configured for '{expected_metric}'. "
                        f"Searches will use the collection's metric '{actual_metric}'."
                    )
            
            self.collection.load()
            logger.info(f"Collection {self.collection_name} loaded")
        except Exception as e:
            logger.error(f"Failed to load collection: {e}")
            raise
    
    def insert(self, file_ids: List[int], embeddings: np.ndarray) -> List[int]:
        """
        Insert embeddings into the collection
        
        Args:
            file_ids: List of file IDs
            embeddings: Numpy array of embeddings (N x dimension)
            
        Returns:
            List of inserted entity IDs
        """
        try:
            if self.collection is None:
                self.collection = Collection(self.collection_name)
            
            # Prepare data
            data = [
                file_ids,
                embeddings.tolist()
            ]
            
            # Insert
            mr = self.collection.insert(data)
            
            # Flush to persist data
            self.collection.flush()
            
            logger.info(f"Inserted {len(file_ids)} embeddings")
            return mr.primary_keys
            
        except Exception as e:
            logger.error(f"Failed to insert embeddings: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10, 
               metric: Optional[SimilarityMetric] = None) -> Tuple[List[int], List[float]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            metric: IGNORED - Milvus uses the metric from index creation. 
                   Only affects query normalization behavior.
            
        Returns:
            Tuple of (file_ids, distances/scores)
        """
        try:
            if self.collection is None:
                self.collection = Collection(self.collection_name)
                self.collection.load()
            
            # IMPORTANT: In Milvus, the metric type is fixed at index creation time.
            # We MUST use the collection's metric, not the requested metric.
            # Detect the actual metric from the collection
            actual_milvus_metric = self.get_collection_metric()
            
            if actual_milvus_metric is None:
                # Fallback to configured metric if detection fails
                milvus_metric = convert_metric_to_milvus(self.metric)
                logger.warning(f"Could not detect collection metric, using configured metric: {milvus_metric}")
            else:
                milvus_metric = actual_milvus_metric
            
            # Warn if trying to use a different metric than collection was created with
            if metric is not None:
                requested_milvus_metric = convert_metric_to_milvus(metric)
                if requested_milvus_metric != milvus_metric:
                    logger.warning(
                        f"Requested metric '{metric.value}' (Milvus: {requested_milvus_metric}) differs from "
                        f"collection metric '{milvus_metric}'. Using collection metric '{milvus_metric}' for search. "
                        f"To use a different metric, recreate the collection with drop_existing=True."
                    )
            
            # Use requested metric for normalization behavior, or default
            search_metric = metric or self.metric
            
            # Normalize for cosine/angular similarity based on requested behavior
            if search_metric in [SimilarityMetric.COSINE, SimilarityMetric.ANGULAR]:
                query_norm = np.linalg.norm(query_embedding) + 1e-8
                query_embedding = query_embedding / query_norm
                logger.debug(f"Normalized query embedding for {search_metric.value}")
            
            # Prepare search params - use collection's metric only
            search_params = {
                "metric_type": milvus_metric,
                "params": {"nprobe": 10}
            }
            
            logger.debug(f"Searching with metric: {milvus_metric} (collection metric: {self.metric.value})")
            
            # Perform search
            query_vector = query_embedding.reshape(1, -1).tolist()
            results = self.collection.search(
                data=query_vector,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["file_id"]
            )
            
            # Extract results
            file_ids = []
            distances = []
            
            if results and len(results) > 0:  # type: ignore
                for hit in results[0]:  # type: ignore
                    file_ids.append(hit.entity.get('file_id'))
                    distances.append(hit.distance)
            
            return file_ids, distances
            
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            raise
    
    def delete(self, file_ids: List[int]):
        """
        Delete embeddings by file IDs
        
        Args:
            file_ids: List of file IDs to delete
        """
        try:
            if self.collection is None:
                self.collection = Collection(self.collection_name)
            
            expr = f"file_id in {file_ids}"
            self.collection.delete(expr)
            self.collection.flush()
            
            logger.info(f"Deleted embeddings for file_ids: {file_ids}")
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            raise
    
    def get_count(self) -> int:
        """Get total number of entities in collection"""
        try:
            if self.collection is None:
                self.collection = Collection(self.collection_name)
            
            self.collection.flush()
            return self.collection.num_entities
            
        except Exception as e:
            logger.error(f"Failed to get count: {e}")
            return 0
    
    def check_health(self) -> Dict[str, Any]:
        """Check Milvus connection health"""
        try:
            if not self.connected:
                return {
                    "connected": False,
                    "error": "Not connected to Milvus"
                }
            
            count = self.get_count()
            
            return {
                "connected": True,
                "collection": self.collection_name,
                "count": count
            }
            
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }
    
    def disconnect(self):
        """Disconnect from Milvus"""
        try:
            connections.disconnect("default")
            self.connected = False
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to disconnect: {e}")


# Global Milvus client instance
_milvus_client = None


def get_milvus_client(metric: Optional[SimilarityMetric] = None) -> Optional[MilvusClient]:
    """
    Get or create global Milvus client instance
    
    Args:
        metric: Similarity metric to use (if creating new instance)
        
    Returns:
        MilvusClient instance or None if connection fails
    """
    global _milvus_client
    if _milvus_client is None:
        try:
            _milvus_client = MilvusClient(metric=metric)
        except Exception as e:
            logger.warning(f"Failed to initialize Milvus client: {e}")
            _milvus_client = None
    return _milvus_client


