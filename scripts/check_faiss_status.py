"""
Check FAISS index status and compatibility

This script checks the FAISS index metric and provides recommendations
for rebuilding if there's a mismatch.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pickle
import faiss
from config import settings

def check_faiss_status():
    """Check FAISS index status"""
    print("\n" + "="*60)
    print("FAISS Index Status")
    print("="*60)
    
    index_path = Path(settings.faiss_index_path)
    metadata_path = Path(settings.faiss_metadata_path)
    
    # Check if files exist
    if not index_path.exists():
        print(f"[ERROR] Index file not found: {index_path}")
        print("   Run: python scripts/ingest_data.py")
        return
    
    if not metadata_path.exists():
        print(f"[ERROR] Metadata file not found: {metadata_path}")
        print("   Run: python scripts/ingest_data.py")
        return
    
    print(f"[OK] Index file: {index_path}")
    print(f"[OK] Metadata file: {metadata_path}")
    
    # Load index
    try:
        index = faiss.read_index(str(index_path))
        print(f"\nIndex Type: {type(index).__name__}")
        print(f"Total vectors: {index.ntotal}")
        print(f"Dimension: {index.d}")
        print(f"Is trained: {index.is_trained}")
    except Exception as e:
        print(f"[ERROR] Failed to load index: {e}")
        return
    
    # Load metadata
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"\nMetadata:")
        print(f"  File IDs: {len(metadata.get('file_ids', []))}")
        print(f"  Dimension: {metadata.get('dimension', 'N/A')}")
        
        metric = metadata.get('metric', 'UNKNOWN')
        print(f"  Metric: {metric}")
        
        # Check compatibility
        print("\n" + "="*60)
        print("Metric Compatibility")
        print("="*60)
        
        index_type = type(index).__name__
        
        if metric == 'l2':
            if 'L2' in index_type:
                print("[OK] Index is L2 (Euclidean distance)")
                print("   Compatible with: similarity_metric=l2")
            else:
                print(f"[WARN] Metric is L2 but index is {index_type}")
        elif metric in ['cosine', 'angular', 'inner_product']:
            if 'IP' in index_type:
                print(f"[OK] Index is Inner Product (for {metric})")
                print(f"   Compatible with: similarity_metric={metric}")
            else:
                print(f"[ERROR] MISMATCH: Metric is {metric} but index is {index_type}")
                print(f"   This will return WRONG results!")
                print(f"\n   To fix, rebuild with:")
                print(f"   python scripts/rebuild_faiss_index.py --metric {metric}")
        
        # Default config check
        print("\n" + "="*60)
        print("Configuration")
        print("="*60)
        print(f"Default metric (config.py): {settings.similarity_metric}")
        
        if settings.similarity_metric != metric:
            print(f"\n[WARN] WARNING: Config metric ({settings.similarity_metric}) != Index metric ({metric})")
            print(f"   Searches without explicit similarity_metric will use: {settings.similarity_metric}")
            print(f"   This may cause errors or wrong results!")
            print(f"\n   Recommended: Rebuild index with config default:")
            print(f"   python scripts/rebuild_faiss_index.py --metric {settings.similarity_metric}")
        else:
            print(f"[OK] Config and index metrics match!")
        
        print("\n" + "="*60)
        print("Recommendations")
        print("="*60)
        
        if metric == 'l2':
            print("Current setup: L2 (Euclidean) distance")
            print("\nPros:")
            print("  - Fast and simple")
            print("  - Works well for many cases")
            print("\nCons:")
            print("  - Sensitive to feature magnitude")
            print("  - Not scale-invariant")
            print("\nFor better results, consider cosine similarity:")
            print("  python scripts/rebuild_faiss_index.py --metric cosine")
        
        elif metric in ['cosine', 'angular']:
            print(f"Current setup: {metric.upper()} similarity")
            print("\nPros:")
            print("  - Scale-invariant (ignores magnitude)")
            print("  - Better for normalized features")
            print("  - Generally better for image similarity")
            print("\nThis is a good choice!")
        
    except Exception as e:
        print(f"[ERROR] Failed to load metadata: {e}")
        return
    
    print("\n" + "="*60)


if __name__ == "__main__":
    check_faiss_status()

