"""Quick check to see if rebuild is progressing"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from app.database import get_db_context
from app.models import LeafImage

print("Checking rebuild progress...\n")

# Check total images
with get_db_context() as db:
    total = db.query(LeafImage).count()
    print(f"Total images in database: {total}")

# Check if index files exist
index_path = Path("data/faiss_index.bin")
metadata_path = Path("data/faiss_metadata.pkl")

print(f"\nIndex file exists: {index_path.exists()}")
print(f"Metadata file exists: {metadata_path.exists()}")

if index_path.exists():
    size_mb = index_path.stat().st_size / (1024 * 1024)
    print(f"Index file size: {size_mb:.2f} MB")

print("\nIf rebuild is running, you should see:")
print("  - Progress bar in the terminal where you started it")
print("  - CPU usage high")
print("  - Process taking 30-45 minutes")
print("\nThe rebuild process is running in background.")
print("You can continue using this terminal for other tasks.")

