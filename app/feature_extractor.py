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
                 use_query_segmentation: bool = False,
                 normalize_after_preprocessing: bool = True):
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
        self.normalize_after_preprocessing = normalize_after_preprocessing # Apply standard normalization after preprocessing
        
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
        # Define the standard ImageNet normalization that ResNet-50 expects
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Resize and crop transforms
        self.resize_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
        # First resize and crop
        if hasattr(self, 'resize_transform'):
            image = self.resize_transform(image)
        else:
            # Fallback to original resize behavior
            image = transforms.Resize(256)(image)
            image = transforms.CenterCrop(224)(image)
        
        # Convert to tensor
        image_tensor = transforms.ToTensor()(image)
        
        # Apply standard ImageNet normalization (critical for ResNet-50)
        if hasattr(self, 'normalize_transform'):
            image_tensor = self.normalize_transform(image_tensor)
        else:
            # Fallback to standard normalization
            image_tensor = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]  # Fixed the std value
            )(image_tensor)
        
        # Apply leaf-specific enhancement to boost contrast and edge detection
        # This helps the model better identify leaf-specific features
        # Enhance the tensor by adjusting the standard deviation to increase feature contrast
        image_tensor = self._enhance_leaf_features(image_tensor)
        
        # Add the leaf feature enhancement method after the existing methods
        if not isinstance(image_tensor, torch.Tensor):
            raise ValueError(f"Transform returned unexpected type: {type(image_tensor)}")
        return image_tensor
    
    def _enhance_leaf_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply leaf-specific feature enhancement to boost important characteristics
        This enhances features that are important for leaf identification
        
        Args:
            image_tensor: Input tensor to enhance
            
        Returns:
            Enhanced tensor
        """
        # Enhance contrast by adjusting the standard deviation of the image
        # This makes the leaf features more prominent
        mean = image_tensor.mean(dim=[1, 2], keepdim=True)  # Compute mean per channel
        std = image_tensor.std(dim=[1, 2], keepdim=True)   # Compute std per channel
        
        # Avoid division by zero
        std = torch.clamp(std, min=1e-8)
        
        # Normalize with adjusted standard deviation to enhance features
        # Use a slightly lower std to increase contrast
        adjusted_std = torch.clamp(std * 1.1, min=1e-8)  # Slightly increase contrast
        image_tensor = (image_tensor - mean) / adjusted_std
        
        # Then renormalize to ImageNet statistics to maintain compatibility with ResNet-50
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        image_tensor = (image_tensor * imagenet_std) + imagenet_mean
        
        return image_tensor
    
    def _additional_normalization(self, features: np.ndarray) -> np.ndarray:
        """
        Apply additional normalization to ensure consistent feature space
        
        Args:
            features: Feature vector to normalize
            
        Returns:
            Normalized feature vector
        """
        # L2 normalize to ensure unit length (most important for cosine similarity)
        feature_norm = np.linalg.norm(features)
        if feature_norm > 0:
            features = features / (feature_norm + 1e-8)
        
        # Then standardize features to have zero mean and unit variance
        # This helps maintain consistency between query and dataset features
        mean = np.mean(features)
        std = np.std(features)
        if std > 0:
            features = (features - mean) / std
            # Renormalize after standardization to ensure unit length
            feature_norm = np.linalg.norm(features)
            if feature_norm > 0:
                features = features / (feature_norm + 1e-8)
        
        return features
    
    def extract_features(self, image_path: Union[str, Path],
                        preprocessing_profile: Optional[PreprocessingProfile] = None,
                        is_query: bool = False,
                        force_normalization: bool = True) -> np.ndarray:
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
            
            # Apply advanced preprocessing consistently for both queries and dataset images
            # Use segmentation for queries and appropriate profile for dataset images
            if self.use_advanced_preprocessing and self.preprocessor:
                # Determine profile based on context
                if is_query and self.use_query_segmentation:
                    # Use FIELD profile for query images which includes aggressive background removal
                    profile = PreprocessingProfile.FIELD
                    logger.info(f"[QUERY] Applying advanced preprocessing with FIELD profile (size: {original_size})")
                else:
                    # Use specified profile for dataset images
                    profile = preprocessing_profile or self.preprocessing_profile
                    logger.debug(f"[DATASET] Applying advanced preprocessing with profile: {profile.value}")
                
                image = self.preprocessor.preprocess(image, profile=profile)
                logger.info(f"[PREPROCESSING] ✓ Advanced preprocessing applied with {profile.value} profile")
            else:
                logger.debug(f"[IMAGE] No preprocessing applied (size: {original_size})")
            
            # Convert to tensor
            image_tensor = self.preprocess_image(image).unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(self.device)
            
            # Extract features
            if self.model is None:
                raise ValueError("Model not initialized")
            
            # Ensure model is in eval mode and extract features
            self.model.eval() # Ensure model is in evaluation mode
            with torch.no_grad():
                features = self.model(image_tensor)
            
            # Convert to numpy and flatten
            features = features.cpu().numpy().flatten()
            
            # L2 normalization (this ensures features are normalized to unit length)
            feature_norm = np.linalg.norm(features)
            features = features / (feature_norm + 1e-8)
            
            # Apply additional normalization if required
            if force_normalization:
                # Additional normalization to ensure consistent feature space
                features = self._additional_normalization(features)
            
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
                
                # Ensure model is in eval mode and extract features
                self.model.eval()  # Ensure model is in evaluation mode
                with torch.no_grad():
                    features = self.model(batch_tensor)
                
                # Convert to numpy
                features = features.cpu().numpy()
                features = features.reshape(features.shape[0], -1)
                
                # L2 normalization
                norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
                features = features / norms
                
                # Apply additional normalization to ensure consistent feature space
                for i in range(features.shape[0]):
                    features[i] = self._additional_normalization(features[i])
                
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
    
    # Use the configuration setting for normalization consistency
    normalize_after = settings.normalize_after_preprocessing
    
    if use_query_segmentation or use_query_preprocessing:
        # Query-specific feature extractor with segmentation and/or preprocessing
        if _query_segmentation_extractor is None:
            _query_segmentation_extractor = FeatureExtractor(
                use_advanced_preprocessing=use_advanced_preprocessing,
                preprocessing_profile=PreprocessingProfile.AUTO,
                use_query_preprocessing=use_query_preprocessing,
                use_query_segmentation=use_query_segmentation,
                normalize_after_preprocessing=normalize_after
            )
        return _query_segmentation_extractor
    elif use_advanced_preprocessing:
        if _advanced_feature_extractor is None:
            _advanced_feature_extractor = FeatureExtractor(
                use_advanced_preprocessing=True,
                preprocessing_profile=PreprocessingProfile.AUTO,
                normalize_after_preprocessing=normalize_after
            )
        return _advanced_feature_extractor
    else:
        if _feature_extractor is None:
            _feature_extractor = FeatureExtractor(normalize_after_preprocessing=normalize_after)
        return _feature_extractor


