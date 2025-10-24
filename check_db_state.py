"""Quick script to check database state"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from app.database import get_db_context
from app.models import LeafImage

with get_db_context() as db:
    total = db.query(LeafImage).count()
    with_faiss = db.query(LeafImage).filter(LeafImage.faiss_id.isnot(None)).count()
    with_milvus = db.query(LeafImage).filter(LeafImage.milvus_id.isnot(None)).count()
    with_both = db.query(LeafImage).filter(
        LeafImage.faiss_id.isnot(None),
        LeafImage.milvus_id.isnot(None)
    ).count()
    with_neither = db.query(LeafImage).filter(
        LeafImage.faiss_id.is_(None),
        LeafImage.milvus_id.is_(None)
    ).count()
    
    print(f"Database State:")
    print(f"  Total images: {total}")
    print(f"  With Faiss ID: {with_faiss}")
    print(f"  With Milvus ID: {with_milvus}")
    print(f"  With both: {with_both}")
    print(f"  With neither: {with_neither}")
    print()
    
    # Check first 10 records
    print("First 10 records:")
    for img in db.query(LeafImage).limit(10):
        print(f"  file_id={img.file_id}, faiss_id={img.faiss_id}, milvus_id={img.milvus_id}")

