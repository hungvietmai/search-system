"""
Test script for database management features
Demonstrates adding new species and uploading images
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import requests
from typing import Optional


class LeafDBManager:
    """Client for database management operations"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def add_species(self, species_name: str, source: str = "lab"):
        """
        Add a new species to the database
        
        Args:
            species_name: Scientific name (e.g., "Quercus montana")
            source: 'lab' or 'field'
        """
        response = requests.post(
            f"{self.base_url}/species/add",
            params={
                'species_name': species_name,
                'source': source
            }
        )
        response.raise_for_status()
        return response.json()
    
    def upload_image(self, species_name: str, image_path: str,
                    segmented_path: Optional[str] = None,
                    source: str = "lab",
                    auto_index: bool = True,
                    search_engine: str = "milvus"):
        """
        Upload an image for a species
        
        Args:
            species_name: Scientific name
            image_path: Path to original image
            segmented_path: Path to segmented image (optional)
            source: 'lab' or 'field'
            auto_index: Auto-extract features and index
            search_engine: 'milvus' or 'faiss'
        """
        files = {
            'original_image': open(image_path, 'rb')
        }
        
        if segmented_path:
            files['segmented_image'] = open(segmented_path, 'rb')
        
        try:
            response = requests.post(
                f"{self.base_url}/species/{species_name}/upload",
                files=files,
                params={
                    'source': source,
                    'auto_index': auto_index,
                    'search_engine': search_engine
                }
            )
            response.raise_for_status()
            return response.json()
        finally:
            # Close file handles
            for f in files.values():
                f.close()
    
    def get_species_stats(self, species_name: str):
        """Get statistics for a species"""
        response = requests.get(
            f"{self.base_url}/species/{species_name}/stats"
        )
        response.raise_for_status()
        return response.json()
    
    def delete_image(self, file_id: int, delete_files: bool = False):
        """
        Delete an image
        
        Args:
            file_id: File ID to delete
            delete_files: Also delete physical files
        """
        response = requests.delete(
            f"{self.base_url}/images/{file_id}",
            params={'delete_files': delete_files}
        )
        response.raise_for_status()
        return response.json()


def main():
    """Main test function"""
    print("=" * 60)
    print("Database Management - Test Script")
    print("=" * 60)
    
    manager = LeafDBManager()
    
    # Test 1: Add a new species
    print("\n1. Adding a new species")
    print("-" * 60)
    
    test_species = "Quercus testus"  # Fictional test species
    
    try:
        result = manager.add_species(test_species, source="lab")
        print(f"✓ {result['message']}")
        print(f"  Folder: {result['folder_name']}")
        print(f"  Images dir: {result['images_directory']}")
        print(f"  Segmented dir: {result['segmented_directory']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Get species stats (before upload)
    print("\n2. Getting species statistics (before upload)")
    print("-" * 60)
    
    try:
        stats = manager.get_species_stats(test_species)
        print(f"Species: {stats['species']}")
        print(f"Total images: {stats['total_images']}")
        print(f"Exists: {stats['exists']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Upload an image (using an existing image for demo)
    print("\n3. Uploading an image")
    print("-" * 60)
    print("Note: This requires an actual image file to upload.")
    print("To test upload, use an existing image from the dataset:")
    print("")
    print("Example:")
    print("  manager = LeafDBManager()")
    print("  result = manager.upload_image(")
    print("      species_name='Quercus testus',")
    print("      image_path='dataset/images/lab/quercus_alba/existing-image.jpg',")
    print("      source='lab',")
    print("      auto_index=True,")
    print("      search_engine='milvus'")
    print("  )")
    
    # Find an existing image to demonstrate
    dataset_path = Path("dataset/images/lab")
    if dataset_path.exists():
        # Find first image from any species
        image_files = list(dataset_path.rglob("*.jpg"))
        if image_files and len(image_files) > 0:
            test_image = image_files[0]
            print(f"\n  Found test image: {test_image}")
            print(f"  Uploading to '{test_species}'...")
            
            try:
                result = manager.upload_image(
                    species_name=test_species,
                    image_path=str(test_image),
                    source="lab",
                    auto_index=True,
                    search_engine="faiss"  # Use Faiss for faster testing
                )
                print(f"  ✓ Upload successful!")
                print(f"    File ID: {result['file_id']}")
                print(f"    Original path: {result['original_path']}")
                print(f"    Indexed: {result['indexed']}")
                if result.get('indexed'):
                    print(f"    Search engine: {result.get('search_engine')}")
                    if 'faiss_id' in result:
                        print(f"    Faiss ID: {result['faiss_id']}")
                
                # Get updated stats
                print("\n4. Getting updated statistics")
                print("-" * 60)
                stats = manager.get_species_stats(test_species)
                print(f"  Total images: {stats['total_images']}")
                print(f"  Lab images: {stats['lab_images']}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    # Test 4: Usage examples
    print("\n" + "=" * 60)
    print("Complete Usage Examples")
    print("=" * 60)
    
    print("\n# Add a new species:")
    print("manager.add_species('Acer newspecies', source='lab')")
    
    print("\n# Upload an image with auto-indexing:")
    print("manager.upload_image(")
    print("    species_name='Acer newspecies',")
    print("    image_path='my_leaf.jpg',")
    print("    source='lab',")
    print("    auto_index=True,")
    print("    search_engine='milvus'")
    print(")")
    
    print("\n# Upload with segmented image:")
    print("manager.upload_image(")
    print("    species_name='Acer newspecies',")
    print("    image_path='my_leaf.jpg',")
    print("    segmented_path='my_leaf_segmented.png',")
    print("    auto_index=True")
    print(")")
    
    print("\n# Get species statistics:")
    print("stats = manager.get_species_stats('Acer newspecies')")
    
    print("\n# Delete an image:")
    print("manager.delete_image(file_id=12345, delete_files=True)")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


