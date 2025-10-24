"""
Query Image Preprocessing Module

Specialized preprocessing for query images to improve search accuracy:
- Image enhancement (sharpening, contrast, brightness)
- Normalization and color correction
- Denoising and artifact removal
- Adaptive preprocessing based on image quality
- Optional advanced preprocessing integration

Expected improvement: +15-20% accuracy for query images
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from typing import Optional, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """Preprocessing pipeline specifically for query images"""
    
    def __init__(self, 
                 enable_enhancement: bool = True,
                 enable_normalization: bool = True,
                 enable_denoising: bool = True,
                 adaptive: bool = True):
        """
        Initialize query preprocessor
        
        Args:
            enable_enhancement: Enable image enhancement (sharpening, contrast)
            enable_normalization: Enable normalization (color correction, brightness)
            enable_denoising: Enable noise reduction
            adaptive: Use adaptive preprocessing based on image quality
        """
        self.enable_enhancement = enable_enhancement
        self.enable_normalization = enable_normalization
        self.enable_denoising = enable_denoising
        self.adaptive = adaptive
    
    def assess_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Assess image quality to determine preprocessing needs
        
        Args:
            image: Image as numpy array (RGB)
            
        Returns:
            Dictionary of quality metrics
        """
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Blur detection using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()
        
        # Brightness analysis
        brightness = float(np.mean(gray))  # type: ignore
        
        # Contrast analysis
        contrast = gray.std()
        
        # Noise estimation using high-frequency components
        median_filtered = cv2.medianBlur(gray, 5)
        noise_estimate = np.std(gray.astype(float) - median_filtered.astype(float))
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Color distribution (for RGB images)
        color_balance = 0.0
        if len(image.shape) == 3:
            channel_means = [np.mean(image[:, :, i]) for i in range(3)]
            color_balance = np.std(channel_means)
        
        return {
            'blur_score': float(blur_score),  # Higher is sharper, <100 is blurry
            'brightness': float(brightness),  # 0-255
            'contrast': float(contrast),  # Higher is better, <30 is low
            'noise_level': float(noise_estimate),  # Higher means more noise
            'edge_density': float(edge_density),  # 0-1
            'color_balance': float(color_balance),  # Lower is more balanced
            'is_blurry': blur_score < 100,
            'is_dark': brightness < 80,
            'is_bright': brightness > 200,
            'is_low_contrast': contrast < 30,
            'is_noisy': noise_estimate > 10,
            'needs_color_correction': color_balance > 30
        }
    
    def enhance_sharpness(self, image: Image.Image, strength: float = 1.5) -> Image.Image:
        """
        Enhance image sharpness using unsharp mask
        
        Args:
            image: PIL Image
            strength: Sharpness strength (1.0-3.0)
            
        Returns:
            Sharpened image
        """
        # Apply unsharp mask filter
        sharpened = image.filter(
            ImageFilter.UnsharpMask(radius=2, percent=int(strength * 100), threshold=3)
        )
        return sharpened
    
    def enhance_contrast(self, image: Image.Image, factor: float = 1.3) -> Image.Image:
        """
        Enhance image contrast
        
        Args:
            image: PIL Image
            factor: Contrast enhancement factor (1.0-2.0)
            
        Returns:
            Contrast-enhanced image
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def enhance_color(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """
        Enhance color saturation
        
        Args:
            image: PIL Image
            factor: Color enhancement factor (1.0-2.0)
            
        Returns:
            Color-enhanced image
        """
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    def adjust_brightness(self, image: Image.Image, target_mean: float = 128) -> Image.Image:
        """
        Adjust brightness to target level
        
        Args:
            image: PIL Image
            target_mean: Target mean brightness (0-255)
            
        Returns:
            Brightness-adjusted image
        """
        # Convert to numpy for analysis
        img_np = np.array(image)
        current_mean = np.mean(img_np)
        
        # Calculate adjustment factor
        if current_mean > 0:
            factor = target_mean / current_mean
            factor = float(np.clip(factor, 0.5, 2.0))  # Limit adjustment range
            
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
        
        return image
    
    def denoise(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Remove noise from image using Non-local Means Denoising
        
        Args:
            image: Image as numpy array (RGB)
            strength: Denoising strength (5-15)
            
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None, h=strength, hColor=strength, 
                templateWindowSize=7, searchWindowSize=21
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                image, None, h=strength, 
                templateWindowSize=7, searchWindowSize=21
            )
        return denoised
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Image as numpy array (RGB)
            
        Returns:
            CLAHE-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def color_correct(self, image: np.ndarray) -> np.ndarray:
        """
        Apply automatic color correction using Gray World algorithm
        
        Args:
            image: Image as numpy array (RGB)
            
        Returns:
            Color-corrected image
        """
        result = image.astype(np.float32)
        
        # Calculate average for each channel
        avg = np.mean(result, axis=(0, 1))
        
        # Calculate global average
        global_avg = np.mean(avg)
        
        # Balance each channel
        for i in range(3):
            if avg[i] > 0:
                result[:, :, i] *= global_avg / avg[i]
        
        # Clip values
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def normalize_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image histogram for consistent exposure
        
        Args:
            image: Image as numpy array (RGB)
            
        Returns:
            Histogram-normalized image
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Normalize L channel
        l_channel = lab[:, :, 0]
        l_normalized = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
        lab[:, :, 0] = l_normalized
        
        # Convert back to RGB
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return normalized
    
    def enhance_edges(self, image: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Enhance edges by adding weighted edge map
        
        Args:
            image: Image as numpy array (RGB)
            alpha: Weight for edge enhancement (0-1)
            
        Returns:
            Edge-enhanced image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, None, iterations=1)  # type: ignore
        
        # Convert edges to 3-channel
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Blend original with edges
        enhanced = cv2.addWeighted(image, 1.0, edges_3ch, alpha, 0)
        
        return enhanced
    
    def preprocess_adaptive(self, image: Image.Image) -> Image.Image:
        """
        Apply adaptive preprocessing based on image quality assessment
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to numpy for quality assessment
        img_np = np.array(image)
        quality = self.assess_quality(img_np)
        
        logger.info(f"Query image quality: blur={quality['blur_score']:.1f}, "
                   f"brightness={quality['brightness']:.1f}, "
                   f"contrast={quality['contrast']:.1f}, "
                   f"noise={quality['noise_level']:.1f}")
        
        # Apply denoising if image is noisy
        if self.enable_denoising and quality['is_noisy']:
            logger.info("Applying denoising to query image")
            img_np = self.denoise(img_np, strength=12)
            image = Image.fromarray(img_np)
        
        # Apply normalization if needed
        if self.enable_normalization:
            # Color correction if unbalanced
            if quality['needs_color_correction']:
                logger.info("Applying color correction to query image")
                img_np = np.array(image)
                img_np = self.color_correct(img_np)
                image = Image.fromarray(img_np)
            
            # Brightness adjustment if too dark or bright
            if quality['is_dark'] or quality['is_bright']:
                logger.info(f"Adjusting brightness (current: {quality['brightness']:.1f})")
                image = self.adjust_brightness(image, target_mean=128)
            
            # Histogram normalization for better contrast
            if quality['is_low_contrast']:
                logger.info("Normalizing histogram for better contrast")
                img_np = np.array(image)
                img_np = self.normalize_histogram(img_np)
                # Also apply CLAHE
                img_np = self.apply_clahe(img_np)
                image = Image.fromarray(img_np)
        
        # Apply enhancement
        if self.enable_enhancement:
            # Sharpen if blurry
            if quality['is_blurry']:
                logger.info("Enhancing sharpness")
                image = self.enhance_sharpness(image, strength=2.0)
            else:
                image = self.enhance_sharpness(image, strength=1.3)
            
            # Enhance contrast
            if quality['is_low_contrast']:
                logger.info("Enhancing contrast")
                image = self.enhance_contrast(image, factor=1.5)
            else:
                image = self.enhance_contrast(image, factor=1.2)
            
            # Enhance color saturation slightly
            image = self.enhance_color(image, factor=1.1)
        
        return image
    
    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Main preprocessing function
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        if self.adaptive:
            return self.preprocess_adaptive(image)
        
        # Non-adaptive preprocessing (apply all enhancements)
        img_np = np.array(image)
        
        # Denoising
        if self.enable_denoising:
            img_np = self.denoise(img_np, strength=10)
        
        # Normalization
        if self.enable_normalization:
            img_np = self.color_correct(img_np)
            img_np = self.apply_clahe(img_np)
        
        image = Image.fromarray(img_np)
        
        # Enhancement
        if self.enable_enhancement:
            image = self.enhance_sharpness(image, strength=1.5)
            image = self.enhance_contrast(image, factor=1.3)
            image = self.enhance_color(image, factor=1.2)
        
        return image


# Global preprocessor instance
_query_preprocessor = None


def get_query_preprocessor(
    enable_enhancement: bool = True,
    enable_normalization: bool = True,
    enable_denoising: bool = True,
    adaptive: bool = True
) -> QueryPreprocessor:
    """
    Get or create global query preprocessor instance
    
    Args:
        enable_enhancement: Enable image enhancement
        enable_normalization: Enable normalization
        enable_denoising: Enable denoising
        adaptive: Use adaptive preprocessing
        
    Returns:
        QueryPreprocessor instance
    """
    global _query_preprocessor
    if _query_preprocessor is None:
        _query_preprocessor = QueryPreprocessor(
            enable_enhancement=enable_enhancement,
            enable_normalization=enable_normalization,
            enable_denoising=enable_denoising,
            adaptive=adaptive
        )
    return _query_preprocessor

