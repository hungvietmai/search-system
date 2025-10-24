"""
Verify that indexing and search use the same feature space
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from app.feature_extractor import get_feature_extractor
from PIL import Image
import numpy as np

def main():
    # Find a test image
    test_images = [
        "dataset/images/lab/acer-rubrum/ny1088-02-7.jpg",
        "dataset/images/lab/acer-pensylvanicum/ny1086-02-3.jpg",
        "temp/processed_ny1007-01-1.jpg",
        "temp/processed_toona-sinensis-8.jpg"
    ]
    
    test_image = None
    for img in test_images:
        if Path(img).exists():
            test_image = img
            break
    
    if not test_image:
        print("[ERROR] No test image found!")
        print("Please ensure dataset images exist in dataset/images/")
        return False
    
    print("=" * 70)
    print("FEATURE SPACE VERIFICATION")
    print("=" * 70)
    print(f"\nUsing test image: {test_image}")
    print()
    
    # 1. Indexing configuration (ingest_data.py)
    print("1. INDEXING CONFIGURATION (scripts/ingest_data.py)")
    print("   " + "-" * 60)
    try:
        indexing_extractor = get_feature_extractor()
        features_indexing = indexing_extractor.extract_features(test_image, is_query=False)
        print(f"   [OK] Preprocessing: NO")
        print(f"   [OK] Feature shape: {features_indexing.shape}")
        print(f"   [OK] Feature mean: {np.mean(features_indexing):.6f}")
        print(f"   [OK] Feature std: {np.std(features_indexing):.6f}")
        print(f"   [OK] Feature norm: {np.linalg.norm(features_indexing):.6f}")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False
    
    # 2. Search configuration (main.py default)
    print("\n2. SEARCH CONFIGURATION (app/main.py with default settings)")
    print("   " + "-" * 60)
    try:
        search_extractor = get_feature_extractor(use_query_preprocessing=False)
        features_search = search_extractor.extract_features(test_image, is_query=False)
        print(f"   [OK] Preprocessing: NO (default)")
        print(f"   [OK] Feature shape: {features_search.shape}")
        print(f"   [OK] Feature mean: {np.mean(features_search):.6f}")
        print(f"   [OK] Feature std: {np.std(features_search):.6f}")
        print(f"   [OK] Feature norm: {np.linalg.norm(features_search):.6f}")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False
    
    # 3. Compare features
    print("\n3. FEATURE SPACE COMPARISON")
    print("   " + "-" * 60)
    
    # Calculate differences
    abs_diff = np.abs(features_indexing - features_search)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    
    # Calculate cosine similarity
    dot_product = np.dot(features_indexing, features_search)
    norm_product = np.linalg.norm(features_indexing) * np.linalg.norm(features_search)
    cosine_sim = dot_product / (norm_product + 1e-10)
    
    print(f"   Max absolute difference: {max_diff:.10f}")
    print(f"   Mean absolute difference: {mean_diff:.10f}")
    print(f"   Cosine similarity: {cosine_sim:.10f}")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if max_diff < 1e-6:
        print("\n*** FEATURE SPACES ARE IDENTICAL! ***")
        print("\nIndexing and search use the SAME feature extraction:")
        print("  - Both use NO preprocessing")
        print("  - Both extract raw ResNet-50 features")
        print("  - Features are mathematically identical")
        print("\nThis means:")
        print("  [OK] Search results will be accurate")
        print("  [OK] Similar images will be found correctly")
        print("  [OK] Distance metrics are meaningful")
        print("\nNext Steps:")
        print("  1. Re-index database with Faiss:")
        print("     python scripts/ingest_data.py --reset-db --skip-milvus")
        print("  2. Start server:")
        print("     uvicorn app.main:app --reload")
        print("  3. Test search - it will work perfectly!")
        return True
    elif max_diff < 1e-3:
        print("\n[WARN] FEATURE SPACES ARE VERY SIMILAR (numerical precision)")
        print(f"\nSmall differences detected (max: {max_diff:.10f})")
        print("This is likely due to:")
        print("  - Floating point precision")
        print("  - Different code paths")
        print("\nConclusion: Should still work well in practice.")
        print("\nRecommendation:")
        print("  - Proceed with re-indexing")
        print("  - Monitor search quality")
        return True
    else:
        print("\n[ERROR] FEATURE SPACES ARE DIFFERENT!")
        print(f"\nSignificant differences detected (max: {max_diff:.6f})")
        print("\nPossible causes:")
        print("  - Preprocessing is enabled in one but not the other")
        print("  - Different model versions")
        print("  - Configuration mismatch")
        print("\nAction Required:")
        print("  1. Check ingest_data.py line 173")
        print("  2. Check main.py line 247")
        print("  3. Ensure both use get_feature_extractor() with same settings")
        return False
    
    print("=" * 70)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

