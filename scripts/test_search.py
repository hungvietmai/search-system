"""
Test script for the Leaf Search System
Demonstrates basic usage of the API
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import requests
from typing import List, Dict
import json


class LeafSearchClient:
    """Simple client for the Leaf Search API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self) -> Dict:
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def search(self, image_path: str, top_k: int = 10, 
               search_engine: str = "milvus", 
               use_segmented: bool = False) -> Dict:
        """
        Search for similar leaves
        
        Args:
            image_path: Path to query image
            top_k: Number of results
            search_engine: 'milvus' or 'faiss'
            use_segmented: Whether to return segmented images
        """
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/search",
                files={'file': f},
                params={
                    'top_k': top_k,
                    'search_engine': search_engine,
                    'use_segmented': use_segmented
                }
            )
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        response = requests.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def list_species(self) -> List[str]:
        """List all species"""
        response = requests.get(f"{self.base_url}/species")
        response.raise_for_status()
        return response.json()
    
    def get_species_images(self, species: str, limit: int = 10) -> List[Dict]:
        """Get images for a species"""
        response = requests.get(
            f"{self.base_url}/species/{species}",
            params={'limit': limit}
        )
        response.raise_for_status()
        return response.json()


def main():
    """Main test function"""
    print("=" * 60)
    print("Leaf Search System - Test Script")
    print("=" * 60)
    
    # Create client
    client = LeafSearchClient()
    
    # 1. Health Check
    print("\n1. Health Check")
    print("-" * 60)
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Milvus Connected: {health['milvus_connected']}")
        print(f"Faiss Loaded: {health['faiss_loaded']}")
        print(f"Total Images: {health['total_images']}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the API server is running: uvicorn app.main:app --reload")
        return
    
    # 2. Statistics
    print("\n2. Database Statistics")
    print("-" * 60)
    try:
        stats = client.get_stats()
        print(f"Total Images: {stats['total_images']}")
        print(f"Total Species: {stats['total_species']}")
        print(f"Lab Images: {stats['total_lab_images']}")
        print(f"Field Images: {stats['total_field_images']}")
        
        print("\nTop 5 Species:")
        for i, sp in enumerate(stats['top_species'][:5], 1):
            print(f"  {i}. {sp['species']}: {sp['total_images']} images")
    except Exception as e:
        print(f"Error: {e}")
    
    # 3. List Species
    print("\n3. Species List (first 10)")
    print("-" * 60)
    try:
        species_list = client.list_species()
        for i, species in enumerate(species_list[:10], 1):
            print(f"  {i}. {species}")
        print(f"\n  ... and {len(species_list) - 10} more species")
    except Exception as e:
        print(f"Error: {e}")
    
    # 4. Search Test (if dataset is available)
    print("\n4. Search Test")
    print("-" * 60)
    print("To test search, run:")
    print("  client = LeafSearchClient()")
    print("  results = client.search('path/to/your/leaf/image.jpg', top_k=5)")
    print("\nExample:")
    print("  results = client.search('dataset/images/lab/acer_rubrum/ny1234-01-1.jpg')")
    
    # Example if you have a test image
    test_image = Path("dataset/images/lab")
    if test_image.exists():
        # Find first image
        image_files = list(test_image.rglob("*.jpg"))
        if image_files:
            print(f"\nRunning search with test image: {image_files[0]}")
            try:
                # Test with Milvus
                print("\nSearching with Milvus...")
                results = client.search(str(image_files[0]), top_k=3, search_engine="milvus")
                print(f"Search Time: {results['search_time_ms']:.2f} ms")
                print(f"Found {results['total_results']} results:")
                for i, result in enumerate(results['results'], 1):
                    print(f"  {i}. {result['species']} (distance: {result['distance']:.4f})")
                
                # Test with Faiss
                print("\nSearching with Faiss...")
                results = client.search(str(image_files[0]), top_k=3, search_engine="faiss")
                print(f"Search Time: {results['search_time_ms']:.2f} ms")
                print(f"Found {results['total_results']} results:")
                for i, result in enumerate(results['results'], 1):
                    print(f"  {i}. {result['species']} (distance: {result['distance']:.4f})")
                    
            except Exception as e:
                print(f"Search error: {e}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


