"""
Feature extraction using ResNet-50 model
Now with advanced preprocessing integration and query-specific preprocessing!
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Union, List, Optional
from pathlib import Path
import logging
from config import settings
from app.preprocessors import get_advanced_preprocessor, PreprocessingProfile
from app.query_preprocessor import get_query_preprocessor

logger = logging.getLogger(__name__)

# Import optimization components lazily to avoid circular imports
def _get_optimized_preprocessor():
    """Lazy import of optimized preprocessor"""
    from app.preprocessing_optimizer import (
        get_optimized_query_preprocessor,
        PreprocessingConfig
    )
    config = PreprocessingConfig(
        enable_caching=settings.enable_preprocessing_cache,
        cache_ttl=settings.preprocessing_cache_ttl,
        enable_parallelization=settings.enable_preprocessing_parallel,
        max_workers=settings.preprocessing_max_workers,
        enable_early_exit=settings.enable_preprocessing_early_exit,
        quality_threshold_blur=settings.preprocessing_quality_threshold_blur,
        quality_threshold_contrast=settings.preprocessing_quality_threshold_contrast
    )
    return get_optimized_query_preprocessor(config=config)


class FeatureExtractor:
    """ResNet-50 based feature extractor for leaf images"""
    
    def __init__(self, model_name: str = "resnet50", use_gpu: bool = True, 
                 use_advanced_preprocessing: bool = False,
                 preprocessing_profile: PreprocessingProfile = PreprocessingProfile.AUTO,
                 use_query_preprocessing: bool = False,
                 use_query_segmentation: bool = False):
        """
        Initialize feature extractor
        
        Args:
            model_name: Name of the model to use
            use_gpu: Whether to use GPU if available
            use_advanced_preprocessing: Whether to use advanced preprocessing
            preprocessing_profile: Preprocessing profile (AUTO, LAB, or FIELD)
            use_query_preprocessing: Whether to use query-specific preprocessing (enhancement, normalization)
            use_query_segmentation: Whether to segment query images to remove backgrounds (uses AdvancedLeafPreprocessor with FIELD profile)
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.use_advanced_preprocessing = use_advanced_preprocessing
        self.preprocessing_profile = preprocessing_profile
        self.use_query_preprocessing = use_query_preprocessing
        self.use_query_segmentation = use_query_segmentation
        
        # Initialize advanced preprocessor if enabled OR if query segmentation is enabled
        if self.use_advanced_preprocessing or self.use_query_segmentation:
            self.preprocessor = get_advanced_preprocessor()
            logger.info(f"Advanced preprocessing ENABLED with profile: {preprocessing_profile.value}")
        else:
            self.preprocessor = None
        
        # Initialize query preprocessor if enabled (for additional enhancement AFTER segmentation)
        if self.use_query_preprocessing:
            # Use optimized preprocessor with caching, parallelization, and early exit
            if settings.enable_preprocessing_cache or settings.enable_preprocessing_early_exit:
                self.query_preprocessor = _get_optimized_preprocessor()
                logger.info("Query preprocessing ENABLED (optimized: caching + early exit + parallelization)")
            else:
                self.query_preprocessor = get_query_preprocessor(
                    enable_enhancement=True,
                    enable_normalization=True,
                    enable_denoising=True,
                    adaptive=True
                )
                logger.info("Query preprocessing ENABLED (enhancement + normalization)")
        else:
            self.query_preprocessor = None
        
        # Log query segmentation status (uses AdvancedLeafPreprocessor with FIELD profile)
        if self.use_query_segmentation:
            logger.info("Query segmentation ENABLED (using AdvancedLeafPreprocessor with aggressive background removal)")

        
        self._load_model()
        self._setup_transforms()
        
        logger.info(f"FeatureExtractor initialized with {model_name} on {self.device}")
    
    def _load_model(self):
        """Load pre-trained ResNet-50 model"""
        try:
            if self.model_name == "resnet50":
                # Load pre-trained ResNet-50
                weights = models.ResNet50_Weights.IMAGENET1K_V2
                self.model = models.resnet50(weights=weights)
                
                # Remove the final classification layer to get features
                # ResNet-50 outputs 2048-dimensional features
                self.model = nn.Sequential(*list(self.model.children())[:-1])
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            # Set to evaluation mode
            self.model.eval()
            self.model.to(self.device)
            
            logger.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load and preprocess an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed tensor
        """
        if self.transform is None:
            raise ValueError("Transform not initialized")
        result = self.transform(image)
        if not isinstance(result, torch.Tensor):
            raise ValueError(f"Transform returned unexpected type: {type(result)}")
        return result
    
    def extract_features(self, image_path: Union[str, Path], 
                        preprocessing_profile: Optional[PreprocessingProfile] = None,
                        is_query: bool = False) -> np.ndarray:
        """
        Extract features from a single image
        
        Args:
            image_path: Path to the image file
            preprocessing_profile: Override default preprocessing profile
            is_query: Whether this is a query image (applies query-specific preprocessing)
            
        Returns:
            Feature vector as numpy array (2048-dimensional for ResNet-50)
        """
        try:
            # Load image
            image = self.load_image(image_path)
            original_size = image.size
            
            # STEP 1: Apply segmentation using AdvancedLeafPreprocessor with FIELD profile for query images
            if is_query and self.use_query_segmentation and self.preprocessor:
                logger.info(f"[QUERY] Applying background removal with FIELD profile (size: {original_size})")
                # Use FIELD profile which includes aggressive background removal
                image = self.preprocessor.preprocess(image, profile=PreprocessingProfile.FIELD)
                logger.info(f"[QUERY] ✓ Background removed using AdvancedLeafPreprocessor")
            
            # STEP 2: Apply query preprocessing (enhancement) if enabled - additional enhancement after segmentation
            elif is_query and self.use_query_preprocessing and self.query_preprocessor:
                logger.info(f"[QUERY] Applying query preprocessing (enhancement + normalization)")
                image = self.query_preprocessor.preprocess(image)
            
            # Apply advanced preprocessing if enabled (for dataset images)
            elif self.use_advanced_preprocessing and self.preprocessor:
                profile = preprocessing_profile or self.preprocessing_profile
                logger.debug(f"[DATASET] Applying preprocessing with profile: {profile.value}")
                image = self.preprocessor.preprocess(image, profile=profile)
            else:
                logger.debug(f"[DATASET] No preprocessing applied (size: {original_size})")
            
            # Convert to tensor
            image_tensor = self.preprocess_image(image).unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(self.device)
            
            # Extract features
            if self.model is None:
                raise ValueError("Model not initialized")
            with torch.no_grad():
                features = self.model(image_tensor)
            
            # Convert to numpy and flatten
            features = features.cpu().numpy().flatten()
            
            # L2 normalization (this ensures features are normalized to unit length)
            feature_norm = np.linalg.norm(features)
            features = features / (feature_norm + 1e-8)
            
            logger.debug(f"Extracted features: shape={features.shape}, norm={np.linalg.norm(features):.4f}, mean={features.mean():.4f}")
            
            return features
        except Exception as e:
            logger.error(f"Failed to extract features from {image_path}: {e}")
            raise
    
    def extract_features_batch(self, image_paths: List[Union[str, Path]], 
                              batch_size: Optional[int] = None) -> np.ndarray:
        """
        Extract features from multiple images in batches
        
        Args:
            image_paths: List of paths to image files
            batch_size: Batch size for processing (default from settings)
            
        Returns:
            Feature matrix as numpy array (N x 2048)
        """
        if batch_size is None:
            batch_size = settings.batch_size
        
        all_features = []
        
        try:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_tensors = []
                
                # Load and preprocess batch
                for path in batch_paths:
                    try:
                        image = self.load_image(path)
                        tensor = self.preprocess_image(image)
                        batch_tensors.append(tensor)
                    except Exception as e:
                        logger.warning(f"Skipping {path}: {e}")
                        continue
                
                if not batch_tensors:
                    continue
                
                # Stack into batch
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                # Extract features
                if self.model is None:
                    raise ValueError("Model not initialized")
                with torch.no_grad():
                    features = self.model(batch_tensor)
                
                # Convert to numpy
                features = features.cpu().numpy()
                features = features.reshape(features.shape[0], -1)
                
                # L2 normalization
                norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
                features = features / norms
                
                all_features.append(features)
            
            # Concatenate all batches
            if all_features:
                return np.vstack(all_features)
            else:
                return np.array([])
                
        except Exception as e:
            logger.error(f"Failed to extract batch features: {e}")
            raise
    
    def get_feature_dim(self) -> int:
        """Get the dimension of feature vectors"""
        return settings.feature_dim


# Global feature extractor instances
_feature_extractor = None
_advanced_feature_extractor = None
_query_feature_extractor = None
_query_segmentation_extractor = None


def get_feature_extractor(use_advanced_preprocessing: bool = False,
                         use_query_preprocessing: bool = False,
                         use_query_segmentation: bool = False) -> FeatureExtractor:
    """
    Get or create global feature extractor instance
    
    Args:
        use_advanced_preprocessing: Whether to use advanced preprocessing
        use_query_preprocessing: Whether to use query-specific preprocessing
        use_query_segmentation: Whether to segment query images (RECOMMENDED for external images)
        
    Returns:
        FeatureExtractor instance
    """
    global _feature_extractor, _advanced_feature_extractor, _query_feature_extractor, _query_segmentation_extractor
    
    if use_query_segmentation or use_query_preprocessing:
        # Query-specific feature extractor with segmentation and/or preprocessing
        if _query_segmentation_extractor is None:
            _query_segmentation_extractor = FeatureExtractor(
                use_advanced_preprocessing=use_advanced_preprocessing,
                preprocessing_profile=PreprocessingProfile.AUTO,
                use_query_preprocessing=use_query_preprocessing,
                use_query_segmentation=use_query_segmentation
            )
        return _query_segmentation_extractor
    elif use_advanced_preprocessing:
        if _advanced_feature_extractor is None:
            _advanced_feature_extractor = FeatureExtractor(
                use_advanced_preprocessing=True,
                preprocessing_profile=PreprocessingProfile.AUTO
            )
        return _advanced_feature_extractor
    else:
        if _feature_extractor is None:
            _feature_extractor = FeatureExtractor()
        return _feature_extractor


