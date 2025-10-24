"""
Upload and file management for new species and images
"""
import shutil
import hashlib
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime
import logging
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)


class UploadManager:
    """Manages file uploads and organization in dataset folder"""
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize upload manager
        
        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path or settings.dataset_path)
        self.images_path = self.dataset_path / "images"
        self.segmented_path = self.dataset_path / "segmented"
    
    @staticmethod
    def normalize_species_name(species_name: str) -> str:
        """
        Normalize species name to folder name format
        
        Examples:
            "Acer rubrum" -> "acer_rubrum"
            "Quercus Robur" -> "quercus_robur"
        
        Args:
            species_name: Scientific name (e.g., "Acer rubrum")
            
        Returns:
            Normalized folder name (e.g., "acer_rubrum")
        """
        # Convert to lowercase and replace spaces with underscores
        normalized = species_name.lower().strip()
        normalized = normalized.replace(" ", "_")
        # Remove any special characters except underscore and hyphen
        normalized = "".join(c for c in normalized if c.isalnum() or c in "_-")
        return normalized
    
    def create_species_directories(self, species_name: str, source: str = "lab") -> Tuple[Path, Path]:
        """
        Create directory structure for a new species
        
        Args:
            species_name: Scientific name
            source: 'lab' or 'field'
            
        Returns:
            Tuple of (images_dir, segmented_dir)
        """
        folder_name = self.normalize_species_name(species_name)
        
        # Create directories
        images_dir = self.images_path / source / folder_name
        segmented_dir = self.segmented_path / source / folder_name
        
        images_dir.mkdir(parents=True, exist_ok=True)
        segmented_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created directories for {species_name} ({folder_name})")
        
        return images_dir, segmented_dir
    
    def generate_filename(self, species_name: str, source: str, 
                         sequence: int, image_number: int, 
                         extension: str = "jpg") -> str:
        """
        Generate filename following the dataset convention
        
        Convention: {identifier}-{sequence:02d}-{image_number}.{ext}
        Example: ny1157-01-1.jpg
        
        Args:
            species_name: Scientific name
            source: 'lab' or 'field'
            sequence: Sequence number (specimen number)
            image_number: Image number for this specimen
            extension: File extension
            
        Returns:
            Generated filename
        """
        # Create a unique identifier based on species and timestamp
        timestamp = datetime.now().strftime("%Y%m")
        folder_name = self.normalize_species_name(species_name)
        
        # Generate identifier (use first letters + timestamp)
        parts = folder_name.split("_")
        initials = "".join(p[0] for p in parts if p)[:4]
        identifier = f"{initials}{timestamp}"
        
        filename = f"{identifier}-{sequence:02d}-{image_number}.{extension}"
        return filename
    
    def get_next_sequence_number(self, species_name: str, source: str) -> int:
        """
        Get the next available sequence number for a species
        
        Args:
            species_name: Scientific name
            source: 'lab' or 'field'
            
        Returns:
            Next sequence number
        """
        folder_name = self.normalize_species_name(species_name)
        images_dir = self.images_path / source / folder_name
        
        if not images_dir.exists():
            return 1
        
        # Find existing files and get max sequence
        max_seq = 0
        for file_path in images_dir.glob("*-*-*.*"):
            try:
                # Parse filename: identifier-seq-num.ext
                parts = file_path.stem.split("-")
                if len(parts) >= 2:
                    seq = int(parts[1])
                    max_seq = max(max_seq, seq)
            except (ValueError, IndexError):
                continue
        
        return max_seq + 1
    
    def save_uploaded_image(self, file_content: bytes, species_name: str, 
                           source: str = "lab", is_segmented: bool = False,
                           sequence: Optional[int] = None) -> Tuple[str, str]:
        """
        Save an uploaded image to the appropriate directory
        
        Args:
            file_content: Image file content (bytes)
            species_name: Scientific name
            source: 'lab' or 'field'
            is_segmented: Whether this is a segmented image
            sequence: Sequence number (auto-generated if None)
            
        Returns:
            Tuple of (relative_image_path, absolute_path)
        """
        # Create directories if they don't exist
        images_dir, segmented_dir = self.create_species_directories(species_name, source)
        
        # Determine sequence number
        if sequence is None:
            sequence = self.get_next_sequence_number(species_name, source)
        
        # Determine target directory and extension
        if is_segmented:
            target_dir = segmented_dir
            extension = "png"
        else:
            target_dir = images_dir
            extension = "jpg"
        
        # Generate filename
        # Count existing images with this sequence
        existing_images = list(target_dir.glob(f"*-{sequence:02d}-*.*"))
        image_number = len(existing_images) + 1
        
        filename = self.generate_filename(species_name, source, sequence, image_number, extension)
        
        # Full path
        file_path = target_dir / filename
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Verify it's a valid image
        try:
            img = Image.open(file_path)
            img.verify()
        except Exception as e:
            # Delete invalid file
            file_path.unlink()
            raise ValueError(f"Invalid image file: {e}")
        
        # Generate relative path (for database)
        if is_segmented:
            relative_path = f"dataset/segmented/{source}/{self.normalize_species_name(species_name)}/{filename}"
        else:
            relative_path = f"dataset/images/{source}/{self.normalize_species_name(species_name)}/{filename}"
        
        logger.info(f"Saved image: {relative_path}")
        
        return relative_path, str(file_path)
    
    def save_image_pair(self, original_content: bytes, segmented_content: Optional[bytes],
                       species_name: str, source: str = "lab") -> Tuple[str, Optional[str]]:
        """
        Save a pair of original and segmented images
        
        Args:
            original_content: Original image content
            segmented_content: Segmented image content (optional)
            species_name: Scientific name
            source: 'lab' or 'field'
            
        Returns:
            Tuple of (original_path, segmented_path)
        """
        # Get sequence number
        sequence = self.get_next_sequence_number(species_name, source)
        
        # Save original
        original_path, _ = self.save_uploaded_image(
            original_content, species_name, source, 
            is_segmented=False, sequence=sequence
        )
        
        # Save segmented if provided
        segmented_path = None
        if segmented_content:
            segmented_path, _ = self.save_uploaded_image(
                segmented_content, species_name, source,
                is_segmented=True, sequence=sequence
            )
        
        return original_path, segmented_path
    
    def validate_species_exists(self, species_name: str, source: str = "lab") -> bool:
        """
        Check if species directory exists
        
        Args:
            species_name: Scientific name
            source: 'lab' or 'field'
            
        Returns:
            True if species exists
        """
        folder_name = self.normalize_species_name(species_name)
        species_dir = self.images_path / source / folder_name
        return species_dir.exists()
    
    def get_species_stats(self, species_name: str) -> dict:
        """
        Get statistics for a species
        
        Args:
            species_name: Scientific name
            
        Returns:
            Dictionary with species statistics
        """
        folder_name = self.normalize_species_name(species_name)
        
        stats = {
            'species': species_name,
            'folder_name': folder_name,
            'lab_images': 0,
            'field_images': 0,
            'lab_segmented': 0,
            'field_segmented': 0
        }
        
        # Count lab images
        lab_dir = self.images_path / "lab" / folder_name
        if lab_dir.exists():
            stats['lab_images'] = len(list(lab_dir.glob("*.jpg")))
        
        # Count field images
        field_dir = self.images_path / "field" / folder_name
        if field_dir.exists():
            stats['field_images'] = len(list(field_dir.glob("*.jpg")))
        
        # Count segmented
        lab_seg_dir = self.segmented_path / "lab" / folder_name
        if lab_seg_dir.exists():
            stats['lab_segmented'] = len(list(lab_seg_dir.glob("*.png")))
        
        field_seg_dir = self.segmented_path / "field" / folder_name
        if field_seg_dir.exists():
            stats['field_segmented'] = len(list(field_seg_dir.glob("*.png")))
        
        return stats


# Global upload manager instance
_upload_manager = None


def get_upload_manager() -> UploadManager:
    """Get or create global upload manager instance"""
    global _upload_manager
    if _upload_manager is None:
        _upload_manager = UploadManager()
    return _upload_manager


