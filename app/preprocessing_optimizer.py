"""
Preprocessing Performance Optimizations

Advanced optimizations for image preprocessing pipeline:
1. Redis-based preprocessing result caching
2. Parallel preprocessing pipeline
3. Early exit conditions for well-processed images
4. Quality-based preprocessing skip logic

Expected improvements:
- Caching: 100-1000x faster for repeated preprocessing
- Parallelization: 3-5x faster for batch operations
- Early exit: 2-3x faster for high-quality images
"""

import numpy as np
import cv2
from PIL import Image
import hashlib
import pickle
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing optimizations"""
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    enable_parallelization: bool = True
    max_workers: int = 4
    enable_early_exit: bool = True
    quality_threshold_blur: float = 200.0  # Skip if sharper
    quality_threshold_contrast: float = 50.0  # Skip if higher contrast
    quality_threshold_brightness_min: float = 100.0  # Skip if in range
    quality_threshold_brightness_max: float = 180.0


class PreprocessingCache:
    """
    Redis-based cache for preprocessing results
    
    Caches preprocessed images to avoid redundant processing.
    Uses image hash as key for fast lookup.
    """
    
    def __init__(self, 
                 use_redis: bool = True,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 1,
                 ttl: int = 3600):
        """
        Initialize preprocessing cache
        
        Args:
            use_redis: Whether to use Redis (falls back to memory)
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database number
            ttl: Default TTL in seconds
        """
        self.ttl = ttl
        self.use_redis = use_redis
        self.cache_backend = None
        self.memory_cache = {}  # Fallback
        
        if use_redis:
            try:
                import redis
                self.cache_backend = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=False
                )
                # Test connection
                self.cache_backend.ping()
                logger.info("Preprocessing Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis not available, using memory cache: {e}")
                self.cache_backend = None
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0
        }
    
    def _hash_image(self, image: Image.Image) -> str:
        """
        Create hash of image for caching
        
        Args:
            image: PIL Image
            
        Returns:
            Hash string
        """
        # Convert to numpy for consistent hashing
        img_array = np.array(image)
        
        # Resize to smaller size for faster hashing
        if img_array.shape[0] > 64 or img_array.shape[1] > 64:
            small = cv2.resize(img_array, (64, 64))
        else:
            small = img_array
        
        # Hash the bytes
        img_bytes = small.tobytes()
        return hashlib.sha256(img_bytes).hexdigest()
    
    def _make_key(self, image_hash: str, profile: str) -> str:
        """Make cache key"""
        return f"preprocess:{profile}:{image_hash}"
    
    def get(self, image: Image.Image, profile: str) -> Optional[Image.Image]:
        """
        Get cached preprocessing result
        
        Args:
            image: Input image (used for hashing)
            profile: Preprocessing profile name
            
        Returns:
            Cached preprocessed image or None
        """
        image_hash = self._hash_image(image)
        key = self._make_key(image_hash, profile)
        
        # Try Redis first
        if self.cache_backend:
            try:
                cached_data = self.cache_backend.get(key)
                if cached_data:
                    # Deserialize
                    data = pickle.loads(cached_data)  # type: ignore
                    # Convert back to PIL Image
                    cached_image = Image.fromarray(data['image_array'])
                    self.stats['hits'] += 1
                    logger.debug(f"Cache HIT for preprocessing: {key[:16]}...")
                    return cached_image
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # Try memory cache
        if key in self.memory_cache:
            cached_image = self.memory_cache[key]
            self.stats['hits'] += 1
            logger.debug(f"Memory cache HIT: {key[:16]}...")
            return cached_image
        
        self.stats['misses'] += 1
        return None
    
    def set(self, 
            original_image: Image.Image,
            preprocessed_image: Image.Image,
            profile: str,
            ttl: Optional[int] = None):
        """
        Cache preprocessing result
        
        Args:
            original_image: Original image (used for hashing)
            preprocessed_image: Preprocessed result to cache
            profile: Preprocessing profile name
            ttl: Time-to-live (None = use default)
        """
        image_hash = self._hash_image(original_image)
        key = self._make_key(image_hash, profile)
        ttl = ttl or self.ttl
        
        # Prepare data
        data = {
            'image_array': np.array(preprocessed_image),
            'timestamp': time.time()
        }
        
        # Store in Redis
        if self.cache_backend:
            try:
                serialized = pickle.dumps(data)
                self.cache_backend.setex(key, ttl, serialized)
                self.stats['sets'] += 1
                logger.debug(f"Cached preprocessing result: {key[:16]}...")
                return
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        # Fallback to memory (with simple size limit)
        if len(self.memory_cache) < 100:  # Limit memory cache size
            self.memory_cache[key] = preprocessed_image
            self.stats['sets'] += 1
    
    def clear(self):
        """Clear all cached preprocessing results"""
        if self.cache_backend:
            try:
                # Clear all preprocessing keys
                pattern = "preprocess:*"
                keys = self.cache_backend.keys(pattern)
                if keys:
                    self.cache_backend.delete(*keys)  # type: ignore
            except Exception as e:
                logger.error(f"Redis clear error: {e}")
        
        self.memory_cache.clear()
        logger.info("Preprocessing cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0.0
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'backend': 'redis' if self.cache_backend else 'memory',
            'memory_cache_size': len(self.memory_cache)
        }


class ImageQualityAnalyzer:
    """
    Analyze image quality to determine if preprocessing is needed
    
    Implements early exit logic for already well-processed images.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize quality analyzer
        
        Args:
            config: Preprocessing configuration with thresholds
        """
        self.config = config
    
    def assess_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        Assess image quality metrics
        
        Args:
            image: Image as numpy array
            
        Returns:
            Dictionary of quality metrics
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()
        
        # Brightness
        brightness = float(np.mean(gray))  # type: ignore
        
        # Contrast
        contrast = gray.std()
        
        # Noise estimation
        median_filtered = cv2.medianBlur(gray, 5)
        noise_level = np.std(gray.astype(float) - median_filtered.astype(float))
        
        return {
            'blur_score': float(blur_score),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'noise_level': float(noise_level)
        }
    
    def should_skip_preprocessing(self, image: Image.Image) -> Tuple[bool, str]:
        """
        Determine if preprocessing can be skipped
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (should_skip, reason)
        """
        if not self.config.enable_early_exit:
            return False, "early_exit_disabled"
        
        img_array = np.array(image)
        quality = self.assess_quality(img_array)
        
        # Check if image is already high quality
        is_sharp = quality['blur_score'] >= self.config.quality_threshold_blur
        is_good_contrast = quality['contrast'] >= self.config.quality_threshold_contrast
        is_good_brightness = (
            self.config.quality_threshold_brightness_min <= quality['brightness'] 
            <= self.config.quality_threshold_brightness_max
        )
        is_clean = quality['noise_level'] < 5.0
        
        if is_sharp and is_good_contrast and is_good_brightness and is_clean:
            reason = f"high_quality (blur={quality['blur_score']:.1f}, contrast={quality['contrast']:.1f})"
            logger.info(f"Early exit: {reason}")
            return True, reason
        
        return False, "needs_preprocessing"
    
    def get_adaptive_params(self, image: Image.Image) -> Dict[str, Any]:
        """
        Get adaptive preprocessing parameters based on quality
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary of preprocessing parameters
        """
        img_array = np.array(image)
        quality = self.assess_quality(img_array)
        
        params = {
            'denoise': quality['noise_level'] > 8.0,
            'denoise_strength': min(15, max(5, int(quality['noise_level']))),
            'sharpen': quality['blur_score'] < 150,
            'sharpen_strength': 2.0 if quality['blur_score'] < 100 else 1.5,
            'enhance_contrast': quality['contrast'] < 40,
            'contrast_factor': 1.5 if quality['contrast'] < 30 else 1.3,
            'adjust_brightness': not (100 <= quality['brightness'] <= 180),
            'brightness_target': 128
        }
        
        return params


class ParallelPreprocessor:
    """
    Parallel preprocessing pipeline for batch operations
    
    Processes multiple images concurrently using thread pool.
    """
    
    def __init__(self, 
                 preprocessor: Any,
                 max_workers: int = 4):
        """
        Initialize parallel preprocessor
        
        Args:
            preprocessor: Base preprocessor instance
            max_workers: Maximum parallel workers
        """
        self.preprocessor = preprocessor
        self.max_workers = max_workers
        logger.info(f"ParallelPreprocessor initialized with {max_workers} workers")
    
    def preprocess_batch(
        self,
        images: List[Image.Image],
        profile: Optional[str] = None
    ) -> List[Image.Image]:
        """
        Preprocess multiple images in parallel
        
        Args:
            images: List of PIL Images
            profile: Preprocessing profile
            
        Returns:
            List of preprocessed images
        """
        if len(images) <= 1:
            # Single image, no need for parallelization
            return [self.preprocessor.preprocess(images[0], profile=profile)] if images else []
        
        results: List[Optional[Image.Image]] = [None] * len(images)
        
        def process_single(idx: int, img: Image.Image) -> Tuple[int, Image.Image]:
            """Process single image"""
            try:
                processed = self.preprocessor.preprocess(img, profile=profile)
                return idx, processed
            except Exception as e:
                logger.error(f"Preprocessing failed for image {idx}: {e}")
                return idx, img  # Return original on error
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_single, i, img): i 
                for i, img in enumerate(images)
            }
            
            for future in as_completed(futures):
                try:
                    idx, processed_img = future.result()
                    results[idx] = processed_img
                except Exception as e:
                    logger.error(f"Error in parallel preprocessing: {e}")
        
        # Filter out None values (shouldn't happen, but just in case)
        return [r for r in results if r is not None]


class OptimizedQueryPreprocessor:
    """
    Query preprocessor with caching, parallelization, and early exit
    
    Wraps the base query preprocessor with optimization layers.
    """
    
    def __init__(self,
                 base_preprocessor: Any,
                 config: Optional[PreprocessingConfig] = None):
        """
        Initialize optimized preprocessor
        
        Args:
            base_preprocessor: Base QueryPreprocessor instance
            config: Optimization configuration
        """
        self.base_preprocessor = base_preprocessor
        self.config = config or PreprocessingConfig()
        
        # Initialize components
        self.cache = PreprocessingCache(
            use_redis=True,
            ttl=self.config.cache_ttl
        ) if self.config.enable_caching else None
        
        self.quality_analyzer = ImageQualityAnalyzer(self.config)
        
        self.parallel_processor = ParallelPreprocessor(
            base_preprocessor,
            max_workers=self.config.max_workers
        ) if self.config.enable_parallelization else None
        
        logger.info("OptimizedQueryPreprocessor initialized")
    
    def preprocess(self, image: Image.Image, profile: str = "default") -> Image.Image:
        """
        Preprocess image with optimizations
        
        Args:
            image: PIL Image
            profile: Preprocessing profile name
            
        Returns:
            Preprocessed image
        """
        start_time = time.time()
        
        # 1. Try cache first
        if self.cache and self.config.enable_caching:
            cached = self.cache.get(image, profile)
            if cached:
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"Preprocessing cache hit ({elapsed:.1f}ms)")
                return cached
        
        # 2. Check if we can skip preprocessing (early exit)
        should_skip, reason = self.quality_analyzer.should_skip_preprocessing(image)
        if should_skip:
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Preprocessing skipped: {reason} ({elapsed:.1f}ms)")
            # Still cache the result (original image)
            if self.cache:
                self.cache.set(image, image, profile)
            return image
        
        # 3. Perform preprocessing
        preprocessed = self.base_preprocessor.preprocess(image)
        
        # 4. Cache the result
        if self.cache and self.config.enable_caching:
            self.cache.set(image, preprocessed, profile)
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Preprocessing completed ({elapsed:.1f}ms)")
        
        return preprocessed
    
    def preprocess_batch(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Preprocess multiple images with parallelization
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of preprocessed images
        """
        if not self.parallel_processor or not self.config.enable_parallelization:
            # Sequential processing
            return [self.preprocess(img) for img in images]
        
        # Parallel processing
        return self.parallel_processor.preprocess_batch(images)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        stats = {
            'config': {
                'caching_enabled': self.config.enable_caching,
                'parallelization_enabled': self.config.enable_parallelization,
                'early_exit_enabled': self.config.enable_early_exit,
                'max_workers': self.config.max_workers
            }
        }
        
        if self.cache:
            stats['cache'] = self.cache.get_stats()
        
        return stats
    
    def clear_cache(self):
        """Clear preprocessing cache"""
        if self.cache:
            self.cache.clear()


# Global optimized preprocessor instance
_optimized_preprocessor = None


def get_optimized_query_preprocessor(
    base_preprocessor: Optional[Any] = None,
    config: Optional[PreprocessingConfig] = None
) -> OptimizedQueryPreprocessor:
    """
    Get or create global optimized query preprocessor
    
    Args:
        base_preprocessor: Base preprocessor instance
        config: Optimization configuration
        
    Returns:
        OptimizedQueryPreprocessor instance
    """
    global _optimized_preprocessor
    
    if _optimized_preprocessor is None:
        if base_preprocessor is None:
            from app.query_preprocessor import get_query_preprocessor
            base_preprocessor = get_query_preprocessor()
        
        _optimized_preprocessor = OptimizedQueryPreprocessor(
            base_preprocessor,
            config=config
        )
    
    return _optimized_preprocessor

