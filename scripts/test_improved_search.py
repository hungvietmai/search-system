"""
Test script to verify the improved search endpoint with trained LTR model
"""
import requests
import os
from pathlib import Path

def test_improved_search():
    """
    Test the improved search endpoint
    """
    # URL for the improved search endpoint
    url = "http://localhost:8000/search-improved"
    
    # Path to a test image (you'll need to update this to an actual image path)
    test_image_path = "test_images/test_leaf.jpg"  # Update this path
    
    # Check if test image exists
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}")
        print("Please provide a valid test image path")
        return False
    
    # Prepare the request
    with open(test_image_path, 'rb') as image_file:
        files = {'file': image_file}
        data = {
            'top_k': 5,
            'use_segmented': False
        }
        
        try:
            print("Sending request to improved search endpoint...")
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                print("✓ Improved search endpoint is working correctly!")
                result = response.json()
                print(f"Found {len(result['results'])} results")
                print(f"Search time: {result['search_time_ms']} ms")
                print(f"Search engine: {result['search_engine']}")
                return True
            else:
                print(f"✗ Request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("✗ Cannot connect to the server. Make sure the FastAPI app is running on localhost:800")
            return False
        except Exception as e:
            print(f"✗ Error during test: {e}")
            return False

def verify_model_exists():
    """
    Verify that the trained model file exists
    """
    model_path = Path("data/trained_ltr_model.pkl")
    if model_path.exists():
        print(f"✓ Trained model found at {model_path}")
        print(f"Model size: {model_path.stat().st_size} bytes")
        return True
    else:
        print(f"✗ Trained model not found at {model_path}")
        print("The endpoint will fall back to basic similarity search if model is not available")
        return False

if __name__ == "__main__":
    print("Testing Improved Search Integration")
    print("="*40)
    
    # Verify model exists
    model_exists = verify_model_exists()
    print()
    
    # Test the endpoint if server is running
    endpoint_works = test_improved_search()
    print()
    
    if model_exists and endpoint_works:
        print("✓ All tests passed! The improved search with LTR model is properly integrated.")
    elif model_exists and not endpoint_works:
        print("⚠ Model exists but endpoint test failed. Check if the server is running.")
    elif not model_exists:
        print("⚠ Model file not found. The endpoint will use fallback behavior.")
    else:
        print("✗ Both model and endpoint tests failed.")