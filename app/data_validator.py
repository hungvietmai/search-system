"""
Data Validation and Quality Checks
Validates incoming images for quality and correctness

Features:
- Image format validation
- Resolution checks
- File corruption detection
- Content validation (leaf detection)
- Quality scoring
- Duplicate detection
- Metadata validation
"""
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
import logging
import imagehash  # type: ignore

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationRule(Enum):
    """Validation rules"""
    FORMAT = "format"
    RESOLUTION = "resolution"
    FILE_SIZE = "file_size"
    CORRUPTION = "corruption"
    CONTENT = "content"
    DUPLICATE = "duplicate"
    METADATA = "metadata"
    QUALITY = "quality"


@dataclass
class ValidationIssue:
    """Validation issue"""
    rule: ValidationRule
    severity: ValidationSeverity
    message: str
    details: Optional[Dict] = None


@dataclass
class ValidationResult:
    """Result of data validation"""
    valid: bool
    score: float  # 0-100
    issues: List[ValidationIssue]
    image_hash: Optional[str] = None
    perceptual_hash: Optional[str] = None
    metrics: Optional[Dict] = None
    
    def is_acceptable(self) -> bool:
        """Check if image is acceptable (no critical/error issues)"""
        return not any(
            issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]
            for issue in self.issues
        )
    
    def get_summary(self) -> Dict:
        """Get validation summary"""
        return {
            'valid': self.valid,
            'acceptable': self.is_acceptable(),
            'score': self.score,
            'total_issues': len(self.issues),
            'critical_issues': sum(1 for i in self.issues if i.severity == ValidationSeverity.CRITICAL),
            'error_issues': sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR),
            'warning_issues': sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING),
            'info_issues': sum(1 for i in self.issues if i.severity == ValidationSeverity.INFO),
            'issues': [
                {
                    'rule': i.rule.value,
                    'severity': i.severity.value,
                    'message': i.message,
                    'details': i.details
                }
                for i in self.issues
            ],
            'metrics': self.metrics or {}
        }


class DataValidator:
    """
    Data validator for leaf images
    Performs comprehensive quality and correctness checks
    """
    
    def __init__(self,
                 min_resolution: Tuple[int, int] = (224, 224),
                 max_resolution: Tuple[int, int] = (4096, 4096),
                 min_file_size: int = 1024,  # 1 KB
                 max_file_size: int = 10 * 1024 * 1024,  # 10 MB
                 allowed_formats: Optional[List[str]] = None,
                 min_quality_score: float = 50.0):
        """
        Initialize data validator
        
        Args:
            min_resolution: Minimum (width, height)
            max_resolution: Maximum (width, height)
            min_file_size: Minimum file size in bytes
            max_file_size: Maximum file size in bytes
            allowed_formats: Allowed image formats
            min_quality_score: Minimum acceptable quality score
        """
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size
        self.allowed_formats = allowed_formats or ['JPEG', 'PNG', 'JPG']
        self.min_quality_score = min_quality_score
        
        # Hash database for duplicate detection
        self.known_hashes: Dict[str, str] = {}  # hash -> file_path
        
        logger.info("Data validator initialized")
    
    def validate(self, image_path: str, species: Optional[str] = None) -> ValidationResult:
        """
        Validate an image
        
        Args:
            image_path: Path to image
            species: Expected species name
            
        Returns:
            ValidationResult
        """
        issues = []
        metrics = {}
        
        path = Path(image_path)
        
        # Check file exists
        if not path.exists():
            return ValidationResult(
                valid=False,
                score=0.0,
                issues=[ValidationIssue(
                    rule=ValidationRule.FORMAT,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"File not found: {image_path}"
                )]
            )
        
        # Check file size
        file_size = path.stat().st_size
        metrics['file_size'] = file_size
        
        if file_size < self.min_file_size:
            issues.append(ValidationIssue(
                rule=ValidationRule.FILE_SIZE,
                severity=ValidationSeverity.ERROR,
                message=f"File too small: {file_size} bytes",
                details={'file_size': file_size, 'min_size': self.min_file_size}
            ))
        
        if file_size > self.max_file_size:
            issues.append(ValidationIssue(
                rule=ValidationRule.FILE_SIZE,
                severity=ValidationSeverity.WARNING,
                message=f"File too large: {file_size} bytes",
                details={'file_size': file_size, 'max_size': self.max_file_size}
            ))
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Check format
            if image.format not in self.allowed_formats:
                issues.append(ValidationIssue(
                    rule=ValidationRule.FORMAT,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid format: {image.format}",
                    details={'format': image.format, 'allowed': self.allowed_formats}
                ))
            
            # Check resolution
            width, height = image.size
            metrics['width'] = width
            metrics['height'] = height
            
            if width < self.min_resolution[0] or height < self.min_resolution[1]:
                issues.append(ValidationIssue(
                    rule=ValidationRule.RESOLUTION,
                    severity=ValidationSeverity.ERROR,
                    message=f"Resolution too low: {width}x{height}",
                    details={'width': width, 'height': height, 'min': self.min_resolution}
                ))
            
            if width > self.max_resolution[0] or height > self.max_resolution[1]:
                issues.append(ValidationIssue(
                    rule=ValidationRule.RESOLUTION,
                    severity=ValidationSeverity.WARNING,
                    message=f"Resolution very high: {width}x{height}",
                    details={'width': width, 'height': height, 'max': self.max_resolution}
                ))
            
            # Check aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            metrics['aspect_ratio'] = aspect_ratio
            
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                issues.append(ValidationIssue(
                    rule=ValidationRule.CONTENT,
                    severity=ValidationSeverity.WARNING,
                    message=f"Unusual aspect ratio: {aspect_ratio:.2f}",
                    details={'aspect_ratio': aspect_ratio}
                ))
            
            # Convert to array for analysis
            img_array = np.array(image.convert('RGB'))
            
            # Check for corruption (all black/white)
            mean_intensity = img_array.mean()
            metrics['mean_intensity'] = float(mean_intensity)
            
            if mean_intensity < 5:
                issues.append(ValidationIssue(
                    rule=ValidationRule.CORRUPTION,
                    severity=ValidationSeverity.ERROR,
                    message="Image appears to be all black",
                    details={'mean_intensity': float(mean_intensity)}
                ))
            
            if mean_intensity > 250:
                issues.append(ValidationIssue(
                    rule=ValidationRule.CORRUPTION,
                    severity=ValidationSeverity.ERROR,
                    message="Image appears to be all white",
                    details={'mean_intensity': float(mean_intensity)}
                ))
            
            # Quality checks
            quality_metrics = self._check_quality(img_array)
            metrics.update(quality_metrics)
            
            # Blur detection
            if quality_metrics.get('blur_score', 100) < 20:
                issues.append(ValidationIssue(
                    rule=ValidationRule.QUALITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Image appears blurry",
                    details={'blur_score': quality_metrics['blur_score']}
                ))
            
            # Brightness check
            if quality_metrics.get('brightness_score', 50) < 20:
                issues.append(ValidationIssue(
                    rule=ValidationRule.QUALITY,
                    severity=ValidationSeverity.WARNING,
                    message="Image too dark",
                    details={'brightness': quality_metrics['brightness_score']}
                ))
            elif quality_metrics.get('brightness_score', 50) > 80:
                issues.append(ValidationIssue(
                    rule=ValidationRule.QUALITY,
                    severity=ValidationSeverity.WARNING,
                    message="Image too bright",
                    details={'brightness': quality_metrics['brightness_score']}
                ))
            
            # Contrast check
            if quality_metrics.get('contrast_score', 50) < 15:
                issues.append(ValidationIssue(
                    rule=ValidationRule.QUALITY,
                    severity=ValidationSeverity.WARNING,
                    message="Low contrast",
                    details={'contrast': quality_metrics['contrast_score']}
                ))
            
            # Content validation (basic leaf detection)
            content_score = self._check_content(img_array)
            metrics['content_score'] = content_score
            
            if content_score < 30:
                issues.append(ValidationIssue(
                    rule=ValidationRule.CONTENT,
                    severity=ValidationSeverity.WARNING,
                    message="Image may not contain a leaf",
                    details={'content_score': content_score}
                ))
            
            # Compute hashes
            image_hash = self._compute_hash(image_path)
            perceptual_hash = str(imagehash.phash(image))
            
            # Duplicate detection
            if image_hash in self.known_hashes:
                issues.append(ValidationIssue(
                    rule=ValidationRule.DUPLICATE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Duplicate of {self.known_hashes[image_hash]}",
                    details={'duplicate_of': self.known_hashes[image_hash]}
                ))
            
            # Metadata validation
            if species:
                if not species.strip():
                    issues.append(ValidationIssue(
                        rule=ValidationRule.METADATA,
                        severity=ValidationSeverity.ERROR,
                        message="Empty species name"
                    ))
                elif len(species) < 3:
                    issues.append(ValidationIssue(
                        rule=ValidationRule.METADATA,
                        severity=ValidationSeverity.WARNING,
                        message=f"Species name very short: '{species}'"
                    ))
            
        except Exception as e:
            logger.error(f"Validation error for {image_path}: {e}")
            issues.append(ValidationIssue(
                rule=ValidationRule.CORRUPTION,
                severity=ValidationSeverity.CRITICAL,
                message=f"Failed to process image: {str(e)}",
                details={'error': str(e)}
            ))
            
            return ValidationResult(
                valid=False,
                score=0.0,
                issues=issues,
                metrics=metrics
            )
        
        # Calculate overall score
        score = self._calculate_score(issues, metrics)
        
        # Determine if valid
        valid = score >= self.min_quality_score and not any(
            i.severity == ValidationSeverity.CRITICAL for i in issues
        )
        
        result = ValidationResult(
            valid=valid,
            score=score,
            issues=issues,
            image_hash=image_hash,
            perceptual_hash=perceptual_hash,
            metrics=metrics
        )
        
        # Store hash if valid
        if valid and image_hash:
            self.known_hashes[image_hash] = image_path
        
        return result
    
    def _check_quality(self, img_array: np.ndarray) -> Dict:
        """
        Check image quality metrics
        
        Args:
            img_array: Image as numpy array
            
        Returns:
            Quality metrics dictionary
        """
        metrics = {}
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['blur_score'] = min(100, float(laplacian_var) / 10)
        
        # Brightness
        brightness = gray.mean()
        metrics['brightness_score'] = float(brightness) / 255 * 100
        
        # Contrast
        contrast = gray.std()
        metrics['contrast_score'] = float(contrast) / 128 * 100
        
        # Noise estimation
        noise = self._estimate_noise(gray)
        metrics['noise_level'] = float(noise)
        
        return metrics
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Estimate image noise level"""
        # Use median absolute deviation
        h, w = gray.shape
        if h < 3 or w < 3:
            return 0.0
        
        # High-pass filter
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Estimate noise
        sigma = np.median(np.abs(filtered - np.median(filtered))) / 0.6745  # type: ignore
        return float(sigma)
    
    def _check_content(self, img_array: np.ndarray) -> float:
        """
        Basic content validation (check if it looks like a leaf)
        
        Args:
            img_array: Image as numpy array
            
        Returns:
            Content score (0-100)
        """
        score = 50.0  # Baseline
        
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Check for green content (common in leaves)
            # Green hue range: 40-80 (in OpenCV 0-180 scale)
            green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            green_ratio = green_mask.sum() / (img_array.shape[0] * img_array.shape[1] * 255)
            
            if green_ratio > 0.2:  # More than 20% green pixels
                score += 30
            elif green_ratio > 0.1:
                score += 15
            
            # Check for edge content (leaves have distinctive edges)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = edges.sum() / (edges.shape[0] * edges.shape[1] * 255)
            
            if 0.05 < edge_ratio < 0.3:  # Reasonable edge density
                score += 20
            
        except Exception as e:
            logger.warning(f"Content check failed: {e}")
        
        return min(100, max(0, score))
    
    def _compute_hash(self, image_path: str) -> str:
        """Compute file hash for duplicate detection"""
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _calculate_score(self, issues: List[ValidationIssue], metrics: Dict) -> float:
        """
        Calculate overall quality score
        
        Args:
            issues: List of validation issues
            metrics: Quality metrics
            
        Returns:
            Score (0-100)
        """
        base_score = 100.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                base_score -= 50
            elif issue.severity == ValidationSeverity.ERROR:
                base_score -= 20
            elif issue.severity == ValidationSeverity.WARNING:
                base_score -= 5
            elif issue.severity == ValidationSeverity.INFO:
                base_score -= 1
        
        # Adjust based on quality metrics
        if 'blur_score' in metrics:
            if metrics['blur_score'] < 30:
                base_score -= 10
        
        if 'content_score' in metrics:
            if metrics['content_score'] < 40:
                base_score -= 15
        
        return max(0, min(100, base_score))
    
    def batch_validate(self, image_paths: List[str]) -> Dict[str, ValidationResult]:
        """
        Validate multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Dictionary mapping paths to validation results
        """
        results = {}
        
        logger.info(f"Validating {len(image_paths)} images")
        
        for path in image_paths:
            try:
                result = self.validate(path)
                results[path] = result
            except Exception as e:
                logger.error(f"Validation failed for {path}: {e}")
                results[path] = ValidationResult(
                    valid=False,
                    score=0.0,
                    issues=[ValidationIssue(
                        rule=ValidationRule.CORRUPTION,
                        severity=ValidationSeverity.CRITICAL,
                        message=str(e)
                    )]
                )
        
        return results
    
    def get_statistics(self, results: Dict[str, ValidationResult]) -> Dict:
        """Get validation statistics"""
        total = len(results)
        valid = sum(1 for r in results.values() if r.valid)
        acceptable = sum(1 for r in results.values() if r.is_acceptable())
        
        avg_score = sum(r.score for r in results.values()) / total if total > 0 else 0
        
        return {
            'total_validated': total,
            'valid': valid,
            'acceptable': acceptable,
            'invalid': total - valid,
            'validation_rate': valid / total * 100 if total > 0 else 0,
            'average_score': avg_score,
            'total_issues': sum(len(r.issues) for r in results.values()),
            'duplicates_found': sum(
                1 for r in results.values()
                for i in r.issues
                if i.rule == ValidationRule.DUPLICATE
            )
        }


# Global validator
_data_validator = None


def get_data_validator() -> DataValidator:
    """
    Get or create global data validator
    
    Returns:
        DataValidator instance
    """
    global _data_validator
    
    if _data_validator is None:
        _data_validator = DataValidator()
    
    return _data_validator

