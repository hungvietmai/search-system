"""
Automated Data Augmentation Pipelines
Generate augmented training data for improved model robustness

Features:
- Geometric transformations (rotation, flip, scale)
- Color augmentations (brightness, contrast, hue, saturation)
- Noise injection
- Blur and sharpening
- Advanced augmentations (cutout, mixup, cutmix)
- Automated pipeline generation
- Stratified augmentation (balance species distribution)
"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import random
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class AugmentationType(Enum):
    """Types of augmentations"""
    ROTATION = "rotation"
    FLIP_HORIZONTAL = "flip_horizontal"
    FLIP_VERTICAL = "flip_vertical"
    SCALE = "scale"
    CROP = "crop"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    HUE = "hue"
    GAUSSIAN_NOISE = "gaussian_noise"
    GAUSSIAN_BLUR = "gaussian_blur"
    SHARPEN = "sharpen"
    CUTOUT = "cutout"
    PERSPECTIVE = "perspective"
    ELASTIC = "elastic"


@dataclass
class AugmentationConfig:
    """Configuration for a single augmentation"""
    aug_type: AugmentationType
    probability: float = 1.0  # 0-1
    params: Optional[Dict] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class AugmentationResult:
    """Result of augmentation"""
    original_path: str
    augmented_image: Image.Image
    augmentations_applied: List[str]
    output_path: Optional[str] = None


class DataAugmentor:
    """
    Data augmentation engine for leaf images
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize data augmentor
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.augmentation_stats = defaultdict(int)
        
        logger.info("Data augmentor initialized")
    
    def augment(self,
                image: Image.Image,
                augmentations: List[AugmentationConfig]) -> Tuple[Image.Image, List[str]]:
        """
        Apply augmentations to an image
        
        Args:
            image: PIL Image
            augmentations: List of augmentation configurations
            
        Returns:
            Tuple of (augmented image, list of applied augmentation names)
        """
        result_image = image.copy()
        applied = []
        
        for aug_config in augmentations:
            # Check probability
            if random.random() > aug_config.probability:
                continue
            
            # Apply augmentation
            try:
                result_image = self._apply_augmentation(
                    result_image,
                    aug_config.aug_type,
                    aug_config.params
                )
                applied.append(aug_config.aug_type.value)
                self.augmentation_stats[aug_config.aug_type.value] += 1
            except Exception as e:
                logger.warning(f"Augmentation {aug_config.aug_type.value} failed: {e}")
        
        return result_image, applied
    
    def _apply_augmentation(self,
                           image: Image.Image,
                           aug_type: AugmentationType,
                           params: Optional[Dict]) -> Image.Image:
        """Apply a single augmentation"""
        if params is None:
            params = {}
        
        if aug_type == AugmentationType.ROTATION:
            angle = params.get('angle', random.uniform(-45, 45))
            return image.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        
        elif aug_type == AugmentationType.FLIP_HORIZONTAL:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        elif aug_type == AugmentationType.FLIP_VERTICAL:
            return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        
        elif aug_type == AugmentationType.SCALE:
            scale = params.get('scale', random.uniform(0.8, 1.2))
            new_size = (int(image.width * scale), int(image.height * scale))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        
        elif aug_type == AugmentationType.CROP:
            crop_ratio = params.get('crop_ratio', 0.9)
            width, height = image.size
            new_width = int(width * crop_ratio)
            new_height = int(height * crop_ratio)
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            return image.crop((left, top, left + new_width, top + new_height))
        
        elif aug_type == AugmentationType.BRIGHTNESS:
            factor = params.get('factor', random.uniform(0.7, 1.3))
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
        
        elif aug_type == AugmentationType.CONTRAST:
            factor = params.get('factor', random.uniform(0.7, 1.3))
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
        
        elif aug_type == AugmentationType.SATURATION:
            factor = params.get('factor', random.uniform(0.7, 1.3))
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(factor)
        
        elif aug_type == AugmentationType.HUE:
            # Convert to HSV, modify hue, convert back
            img_array = np.array(image.convert('RGB'))
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
            hue_shift = params.get('shift', random.uniform(-20, 20))
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            return Image.fromarray(rgb)
        
        elif aug_type == AugmentationType.GAUSSIAN_NOISE:
            img_array = np.array(image).astype(np.float32)
            noise_std = params.get('std', 10)
            noise = np.random.normal(0, noise_std, img_array.shape)
            noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy)
        
        elif aug_type == AugmentationType.GAUSSIAN_BLUR:
            radius = params.get('radius', random.uniform(0.5, 2.0))
            return image.filter(ImageFilter.GaussianBlur(radius))
        
        elif aug_type == AugmentationType.SHARPEN:
            return image.filter(ImageFilter.SHARPEN)
        
        elif aug_type == AugmentationType.CUTOUT:
            img_array = np.array(image).copy()
            h, w = img_array.shape[:2]
            cutout_size = params.get('size', min(h, w) // 8)
            x = random.randint(0, w - cutout_size)
            y = random.randint(0, h - cutout_size)
            img_array[y:y+cutout_size, x:x+cutout_size] = 128  # Gray
            return Image.fromarray(img_array)
        
        elif aug_type == AugmentationType.PERSPECTIVE:
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            
            # Define source points (corners)
            src_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            
            # Define destination points with random perturbation
            max_offset = params.get('max_offset', min(w, h) * 0.1)
            dst_points = src_points + np.random.uniform(-max_offset, max_offset, src_points.shape).astype(np.float32)
            
            # Compute perspective transform
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            warped = cv2.warpPerspective(img_array, matrix, (w, h), 
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(255, 255, 255))
            return Image.fromarray(warped)
        
        elif aug_type == AugmentationType.ELASTIC:
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            
            alpha = params.get('alpha', 50)
            sigma = params.get('sigma', 5)
            
            # Generate random displacement fields
            dx = np.random.randn(h, w) * sigma
            dy = np.random.randn(h, w) * sigma
            
            # Smooth the fields
            dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
            dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
            
            # Create meshgrid
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x_new = (x + dx).astype(np.float32)
            y_new = (y + dy).astype(np.float32)
            
            # Apply displacement
            distorted = cv2.remap(img_array, x_new, y_new,
                                 cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
            return Image.fromarray(distorted)
        
        return image
    
    def create_pipeline(self, profile: str = "standard") -> List[AugmentationConfig]:
        """
        Create a predefined augmentation pipeline
        
        Args:
            profile: Pipeline profile (standard, aggressive, minimal)
            
        Returns:
            List of augmentation configurations
        """
        if profile == "minimal":
            return [
                AugmentationConfig(AugmentationType.ROTATION, 0.5, {'angle': None}),
                AugmentationConfig(AugmentationType.FLIP_HORIZONTAL, 0.5),
                AugmentationConfig(AugmentationType.BRIGHTNESS, 0.3, {'factor': None}),
            ]
        
        elif profile == "aggressive":
            return [
                AugmentationConfig(AugmentationType.ROTATION, 0.7, {'angle': None}),
                AugmentationConfig(AugmentationType.FLIP_HORIZONTAL, 0.5),
                AugmentationConfig(AugmentationType.FLIP_VERTICAL, 0.3),
                AugmentationConfig(AugmentationType.SCALE, 0.5, {'scale': None}),
                AugmentationConfig(AugmentationType.BRIGHTNESS, 0.6, {'factor': None}),
                AugmentationConfig(AugmentationType.CONTRAST, 0.6, {'factor': None}),
                AugmentationConfig(AugmentationType.SATURATION, 0.4, {'factor': None}),
                AugmentationConfig(AugmentationType.HUE, 0.3, {'shift': None}),
                AugmentationConfig(AugmentationType.GAUSSIAN_NOISE, 0.3, {'std': 5}),
                AugmentationConfig(AugmentationType.GAUSSIAN_BLUR, 0.2, {'radius': None}),
                AugmentationConfig(AugmentationType.CUTOUT, 0.3, {'size': None}),
                AugmentationConfig(AugmentationType.PERSPECTIVE, 0.3),
                AugmentationConfig(AugmentationType.ELASTIC, 0.2),
            ]
        
        else:  # standard
            return [
                AugmentationConfig(AugmentationType.ROTATION, 0.6, {'angle': None}),
                AugmentationConfig(AugmentationType.FLIP_HORIZONTAL, 0.5),
                AugmentationConfig(AugmentationType.SCALE, 0.3, {'scale': None}),
                AugmentationConfig(AugmentationType.BRIGHTNESS, 0.4, {'factor': None}),
                AugmentationConfig(AugmentationType.CONTRAST, 0.4, {'factor': None}),
                AugmentationConfig(AugmentationType.SATURATION, 0.3, {'factor': None}),
                AugmentationConfig(AugmentationType.GAUSSIAN_NOISE, 0.2, {'std': 5}),
                AugmentationConfig(AugmentationType.GAUSSIAN_BLUR, 0.2, {'radius': 1.0}),
            ]
    
    def augment_dataset(self,
                       image_paths: List[str],
                       output_dir: str,
                       augmentations_per_image: int = 5,
                       pipeline: Optional[List[AugmentationConfig]] = None,
                       prefix: str = "aug_") -> List[AugmentationResult]:
        """
        Augment an entire dataset
        
        Args:
            image_paths: List of image paths
            output_dir: Output directory for augmented images
            augmentations_per_image: Number of augmented versions per image
            pipeline: Augmentation pipeline (uses standard if None)
            prefix: Prefix for augmented image filenames
            
        Returns:
            List of augmentation results
        """
        if pipeline is None:
            pipeline = self.create_pipeline("standard")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        logger.info(f"Augmenting {len(image_paths)} images, "
                   f"{augmentations_per_image} versions each")
        
        for img_path in image_paths:
            try:
                # Load image
                image = Image.open(img_path)
                
                # Generate augmented versions
                for i in range(augmentations_per_image):
                    aug_image, applied = self.augment(image, pipeline)
                    
                    # Save augmented image
                    filename = Path(img_path).stem
                    ext = Path(img_path).suffix
                    output_file = output_path / f"{prefix}{filename}_{i}{ext}"
                    aug_image.save(output_file)
                    
                    results.append(AugmentationResult(
                        original_path=img_path,
                        augmented_image=aug_image,
                        augmentations_applied=applied,
                        output_path=str(output_file)
                    ))
                
            except Exception as e:
                logger.error(f"Augmentation failed for {img_path}: {e}")
        
        logger.info(f"Generated {len(results)} augmented images")
        
        return results
    
    def stratified_augmentation(self,
                               species_images: Dict[str, List[str]],
                               output_dir: str,
                               target_count: int = 100,
                               pipeline: Optional[List[AugmentationConfig]] = None) -> Dict[str, List[str]]:
        """
        Perform stratified augmentation to balance species distribution
        
        Args:
            species_images: Dictionary mapping species to image paths
            output_dir: Output directory
            target_count: Target number of images per species
            pipeline: Augmentation pipeline
            
        Returns:
            Dictionary mapping species to augmented image paths
        """
        if pipeline is None:
            pipeline = self.create_pipeline("standard")
        
        output_path = Path(output_dir)
        augmented_paths = defaultdict(list)
        
        logger.info(f"Performing stratified augmentation, target: {target_count} per species")
        
        for species, image_paths in species_images.items():
            current_count = len(image_paths)
            
            # Keep original images
            augmented_paths[species].extend(image_paths)
            
            if current_count >= target_count:
                logger.info(f"{species}: {current_count} images (no augmentation needed)")
                continue
            
            # Calculate how many augmented versions needed
            needed = target_count - current_count
            augs_per_image = needed // current_count + 1
            
            logger.info(f"{species}: {current_count} -> {target_count} "
                       f"({augs_per_image} augmentations per image)")
            
            # Create species output directory
            species_dir = output_path / species
            species_dir.mkdir(parents=True, exist_ok=True)
            
            # Augment
            results = self.augment_dataset(
                image_paths,
                str(species_dir),
                augmentations_per_image=augs_per_image,
                pipeline=pipeline,
                prefix=f"{species}_aug_"
            )
            
            # Add augmented paths
            augmented_paths[species].extend([r.output_path for r in results if r.output_path])
            
            # Trim to target count
            augmented_paths[species] = augmented_paths[species][:target_count]
        
        return dict(augmented_paths)
    
    def mixup(self,
             image1: Image.Image,
             image2: Image.Image,
             alpha: float = 0.5) -> Image.Image:
        """
        Apply MixUp augmentation (blend two images)
        
        Args:
            image1: First image
            image2: Second image
            alpha: Blending factor (0-1)
            
        Returns:
            Mixed image
        """
        # Ensure same size
        if image1.size != image2.size:
            image2 = image2.resize(image1.size, Image.Resampling.LANCZOS)
        
        # Blend
        return Image.blend(image1, image2, alpha)
    
    def cutmix(self,
               image1: Image.Image,
               image2: Image.Image,
               cutout_ratio: float = 0.5) -> Image.Image:
        """
        Apply CutMix augmentation (paste region from image2 into image1)
        
        Args:
            image1: Base image
            image2: Source image for cutout
            cutout_ratio: Ratio of area to cut (0-1)
            
        Returns:
            CutMix image
        """
        # Ensure same size
        if image1.size != image2.size:
            image2 = image2.resize(image1.size, Image.Resampling.LANCZOS)
        
        img1_array = np.array(image1)
        img2_array = np.array(image2)
        
        h, w = img1_array.shape[:2]
        
        # Calculate cutout size
        cutout_w = int(w * np.sqrt(cutout_ratio))
        cutout_h = int(h * np.sqrt(cutout_ratio))
        
        # Random position
        x = random.randint(0, w - cutout_w)
        y = random.randint(0, h - cutout_h)
        
        # Copy region
        img1_array[y:y+cutout_h, x:x+cutout_w] = img2_array[y:y+cutout_h, x:x+cutout_w]
        
        return Image.fromarray(img1_array)
    
    def get_statistics(self) -> Dict:
        """Get augmentation statistics"""
        total = sum(self.augmentation_stats.values())
        
        return {
            'total_augmentations': total,
            'augmentations_by_type': dict(self.augmentation_stats),
            'most_used': max(self.augmentation_stats.items(), key=lambda x: x[1])[0] if self.augmentation_stats else None
        }


# Global augmentor
_data_augmentor = None


def get_data_augmentor(seed: Optional[int] = None) -> DataAugmentor:
    """
    Get or create global data augmentor
    
    Args:
        seed: Random seed
        
    Returns:
        DataAugmentor instance
    """
    global _data_augmentor
    
    if _data_augmentor is None:
        _data_augmentor = DataAugmentor(seed=seed)
    
    return _data_augmentor

