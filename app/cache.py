"""
Caching Layer for Performance Optimization
Supports Redis and in-memory caching with automatic fallback

Features:
- Feature vector caching
- Search result caching
- Query hash caching
- TTL (Time-To-Live) support
- Cache invalidation
- Statistics tracking
"""
import numpy as np
import pickle
import hashlib
import json
from typing import Optional, Any, Dict, List, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
import logging
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


class CacheBackend(Enum):
    """Cache backend types"""
    REDIS = "redis"
    MEMORY = "memory"
    HYBRID = "hybrid"


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'deletes': self.deletes,
            'size': self.size,
            'hit_rate': self.hit_rate
        }


class LRUCache:
    """Thread-safe LRU (Least Recently Used) cache"""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache
        
        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.stats.hits += 1
                return self.cache[key]
            else:
                self.stats.misses += 1
                return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
            else:
                # Add new
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    self.cache.popitem(last=False)
                
                self.stats.sets += 1
            
            self.cache[key] = value
            self.stats.size = len(self.cache)
    
    def delete(self, key: str):
        """Delete value from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats.deletes += 1
                self.stats.size = len(self.cache)
    
    def clear(self):
        """Clear entire cache"""
        with self.lock:
            self.cache.clear()
            self.stats.size = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.stats


class RedisCache:
    """Redis-based cache with automatic serialization"""
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 prefix: str = 'leaf_search:'):
        """
        Initialize Redis cache
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            prefix: Key prefix for namespacing
        """
        self.prefix = prefix
        self.stats = CacheStats()
        
        try:
            import redis
            self.redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False  # Binary mode for pickle
            )
            # Test connection
            self.redis.ping()
            self.available = True
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis = None
            self.available = False
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key"""
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        if not self.available or self.redis is None:
            return None
        
        try:
            full_key = self._make_key(key)
            value = self.redis.get(full_key)
            
            if value is not None:
                self.stats.hits += 1
                # decode_responses=False ensures value is bytes
                return pickle.loads(value)  # type: ignore[arg-type]
            else:
                self.stats.misses += 1
                return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in Redis
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = no expiration)
        """
        if not self.available or self.redis is None:
            return
        
        try:
            full_key = self._make_key(key)
            serialized = pickle.dumps(value)
            
            if ttl:
                self.redis.setex(full_key, ttl, serialized)
            else:
                self.redis.set(full_key, serialized)
            
            self.stats.sets += 1
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def delete(self, key: str):
        """Delete value from Redis"""
        if not self.available or self.redis is None:
            return
        
        try:
            full_key = self._make_key(key)
            deleted = self.redis.delete(full_key)
            if deleted:
                self.stats.deletes += 1
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
    
    def clear(self, pattern: str = "*"):
        """
        Clear cache keys matching pattern
        
        Args:
            pattern: Key pattern (default: all keys with prefix)
        """
        if not self.available or self.redis is None:
            return
        
        try:
            full_pattern = self._make_key(pattern)
            keys = self.redis.keys(full_pattern)
            if keys:
                self.redis.delete(*keys)  # type: ignore[arg-type]
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        if self.available and self.redis is not None:
            try:
                info = self.redis.info('stats')
                # Update size from Redis
                self.stats.size = int(self.redis.dbsize())  # type: ignore[arg-type]
            except:
                pass
        
        return self.stats


class HybridCache:
    """
    Hybrid cache using both Redis and in-memory
    L1: In-memory (fast, small)
    L2: Redis (larger, persistent)
    """
    
    def __init__(self,
                 memory_size: int = 100,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0):
        """
        Initialize hybrid cache
        
        Args:
            memory_size: Size of in-memory cache
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database number
        """
        # L1 cache (in-memory)
        self.l1_cache = LRUCache(max_size=memory_size)
        
        # L2 cache (Redis)
        self.l2_cache = RedisCache(
            host=redis_host,
            port=redis_port,
            db=redis_db
        )
        
        logger.info(f"Hybrid cache initialized (L1: {memory_size} items, L2: Redis)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 first, then L2)"""
        # Try L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Try L2 cache
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self.l1_cache.set(key, value)
            return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in both caches"""
        # Set in L1
        self.l1_cache.set(key, value)
        
        # Set in L2 with TTL
        self.l2_cache.set(key, value, ttl)
    
    def delete(self, key: str):
        """Delete from both caches"""
        self.l1_cache.delete(key)
        self.l2_cache.delete(key)
    
    def clear(self):
        """Clear both caches"""
        self.l1_cache.clear()
        self.l2_cache.clear()
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for both caches"""
        return {
            'l1': self.l1_cache.get_stats(),
            'l2': self.l2_cache.get_stats()
        }


class FeatureVectorCache:
    """Specialized cache for feature vectors"""
    
    def __init__(self, backend: Optional[Any] = None):
        """
        Initialize feature vector cache
        
        Args:
            backend: Cache backend (LRU, Redis, or Hybrid)
        """
        self.backend = backend or LRUCache(max_size=1000)
        logger.info(f"Feature vector cache initialized with {type(self.backend).__name__}")
    
    def _make_key(self, file_id: int) -> str:
        """Make cache key for file ID"""
        return f"feature:{file_id}"
    
    def get_features(self, file_id: int) -> Optional[np.ndarray]:
        """Get feature vector for file ID"""
        key = self._make_key(file_id)
        value = self.backend.get(key)
        
        if value is not None:
            return np.array(value)
        
        return None
    
    def set_features(self, file_id: int, features: np.ndarray, ttl: int = 3600):
        """
        Cache feature vector
        
        Args:
            file_id: File ID
            features: Feature vector
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        key = self._make_key(file_id)
        # Convert to list for serialization
        value = features.tolist()
        
        # Try to set with TTL, fall back to without if not supported
        try:
            if hasattr(self.backend, 'set'):
                self.backend.set(key, value, ttl=ttl)  # type: ignore
        except TypeError:
            # Backend doesn't support TTL parameter
            self.backend.set(key, value)
    
    def delete_features(self, file_id: int):
        """Delete feature vector from cache"""
        key = self._make_key(file_id)
        self.backend.delete(key)
    
    def get_stats(self):
        """Get cache statistics"""
        return self.backend.get_stats()


class SearchResultCache:
    """Cache for search results"""
    
    def __init__(self, backend: Optional[Any] = None, ttl: int = 300):
        """
        Initialize search result cache
        
        Args:
            backend: Cache backend
            ttl: Default TTL in seconds (default: 5 minutes)
        """
        self.backend = backend or LRUCache(max_size=500)
        self.default_ttl = ttl
        logger.info(f"Search result cache initialized with {type(self.backend).__name__}")
    
    def _make_key(self, query_hash: str, top_k: int, engine: str) -> str:
        """Make cache key for search query"""
        return f"search:{query_hash}:{top_k}:{engine}"
    
    def _hash_query(self, query_features: np.ndarray) -> str:
        """Hash query features to create key"""
        # Use first 100 dimensions for hashing (faster)
        features_subset = query_features[:100] if len(query_features) > 100 else query_features
        feature_bytes = features_subset.tobytes()
        return hashlib.md5(feature_bytes).hexdigest()
    
    def get_results(self,
                   query_features: np.ndarray,
                   top_k: int,
                   engine: str) -> Optional[Tuple[List[int], List[float]]]:
        """
        Get cached search results
        
        Args:
            query_features: Query feature vector
            top_k: Number of results
            engine: Search engine name
            
        Returns:
            Tuple of (file_ids, distances) or None
        """
        query_hash = self._hash_query(query_features)
        key = self._make_key(query_hash, top_k, engine)
        
        result = self.backend.get(key)
        if result:
            return result['file_ids'], result['distances']
        
        return None
    
    def set_results(self,
                   query_features: np.ndarray,
                   top_k: int,
                   engine: str,
                   file_ids: List[int],
                   distances: List[float],
                   ttl: Optional[int] = None):
        """
        Cache search results
        
        Args:
            query_features: Query feature vector
            top_k: Number of results
            engine: Search engine name
            file_ids: Result file IDs
            distances: Result distances
            ttl: Time-to-live (None = use default)
        """
        query_hash = self._hash_query(query_features)
        key = self._make_key(query_hash, top_k, engine)
        
        value = {
            'file_ids': file_ids,
            'distances': distances,
            'cached_at': datetime.now(timezone.utc).isoformat()
        }
        
        ttl = ttl or self.default_ttl
        
        # Try to set with TTL, fall back to without if not supported
        try:
            if hasattr(self.backend, 'set'):
                self.backend.set(key, value, ttl=ttl)  # type: ignore
        except TypeError:
            # Backend doesn't support TTL parameter
            self.backend.set(key, value)
    
    def invalidate_all(self):
        """Invalidate all cached search results"""
        if hasattr(self.backend, 'clear'):
            self.backend.clear()
    
    def get_stats(self):
        """Get cache statistics"""
        return self.backend.get_stats()


# Global cache instances
_feature_cache = None
_search_cache = None
_hybrid_cache = None


def get_feature_cache() -> FeatureVectorCache:
    """Get or create global feature vector cache"""
    global _feature_cache, _hybrid_cache
    
    if _feature_cache is None:
        # Try to use hybrid cache if available
        if _hybrid_cache is None:
            _hybrid_cache = HybridCache(memory_size=100)
        
        _feature_cache = FeatureVectorCache(backend=_hybrid_cache)
    
    return _feature_cache


def get_search_cache() -> SearchResultCache:
    """Get or create global search result cache"""
    global _search_cache, _hybrid_cache
    
    if _search_cache is None:
        # Try to use hybrid cache if available
        if _hybrid_cache is None:
            _hybrid_cache = HybridCache(memory_size=100)
        
        _search_cache = SearchResultCache(backend=_hybrid_cache, ttl=300)
    
    return _search_cache


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches"""
    stats = {}
    
    if _feature_cache:
        stats['feature_cache'] = _feature_cache.get_stats()
    
    if _search_cache:
        stats['search_cache'] = _search_cache.get_stats()
    
    if _hybrid_cache:
        hybrid_stats = _hybrid_cache.get_stats()
        stats['hybrid_cache'] = {
            'l1': hybrid_stats['l1'].to_dict() if hasattr(hybrid_stats['l1'], 'to_dict') else str(hybrid_stats['l1']),
            'l2': hybrid_stats['l2'].to_dict() if hasattr(hybrid_stats['l2'], 'to_dict') else str(hybrid_stats['l2'])
        }
    
    return stats


def clear_all_caches():
    """Clear all caches"""
    if _feature_cache:
        _feature_cache.backend.clear()
    
    if _search_cache:
        _search_cache.backend.clear()
    
    if _hybrid_cache:
        _hybrid_cache.clear()
    
    logger.info("All caches cleared")

