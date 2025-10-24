"""
Test script for Data Management features

Demonstrates:
1. Data validation
2. Image augmentation
3. Incremental indexing
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data_validator import get_data_validator
from app.data_augmentation import get_data_augmentor, AugmentationType, AugmentationConfig
from app.incremental_indexer import get_incremental_indexer
from PIL import Image
import numpy as np


def test_validation():
    """Test image validation"""
    print("\n" + "="*70)
    print("TEST 1: DATA VALIDATION")
    print("="*70)
    
    # Create test image
    test_image_path = "test_validation_image.jpg"
    
    # Create a simple test image (green leaf-like)
    img_array = np.ones((512, 512, 3), dtype=np.uint8) * 255
    # Add green region (simulating leaf)
    img_array[100:400, 100:400, 0] = 50   # R
    img_array[100:400, 100:400, 1] = 200  # G
    img_array[100:400, 100:400, 2] = 50   # B
    
    img = Image.fromarray(img_array)
    img.save(test_image_path, quality=95)
    
    # Validate
    validator = get_data_validator()
    result = validator.validate(test_image_path, species="test_maple")
    
    print(f"\n✓ Image validated")
    print(f"  Valid: {result.valid}")
    print(f"  Acceptable: {result.is_acceptable()}")
    print(f"  Score: {result.score:.1f}/100")
    
    print(f"\n  Metrics:")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"    - {key}: {value:.2f}")
        else:
            print(f"    - {key}: {value}")
    
    if result.issues:
        print(f"\n  Issues found: {len(result.issues)}")
        for issue in result.issues[:3]:  # Show first 3
            print(f"    [{issue.severity.value}] {issue.message}")
    else:
        print("\n  No issues found!")
    
    # Cleanup
    Path(test_image_path).unlink()
    
    print("\n✅ Validation test complete")


def test_augmentation():
    """Test image augmentation"""
    print("\n" + "="*70)
    print("TEST 2: DATA AUGMENTATION")
    print("="*70)
    
    # Create test image
    test_image_path = "test_aug_image.jpg"
    img_array = np.ones((256, 256, 3), dtype=np.uint8) * 255
    img_array[50:200, 50:200] = [100, 150, 100]  # Green square
    
    img = Image.fromarray(img_array)
    img.save(test_image_path)
    
    # Test augmentation
    augmentor = get_data_augmentor(seed=42)
    
    print("\n✓ Testing augmentation profiles:")
    
    for profile_name in ["minimal", "standard", "aggressive"]:
        pipeline = augmentor.create_pipeline(profile_name)
        print(f"\n  {profile_name.upper()} Profile:")
        print(f"    - {len(pipeline)} augmentation types")
        
        # Apply augmentation
        augmented, applied = augmentor.augment(img, pipeline)
        print(f"    - Applied: {', '.join(applied[:5])}")
        if len(applied) > 5:
            print(f"      ... and {len(applied) - 5} more")
    
    # Test custom pipeline
    print("\n✓ Testing custom pipeline:")
    custom_pipeline = [
        AugmentationConfig(AugmentationType.ROTATION, probability=1.0),
        AugmentationConfig(AugmentationType.BRIGHTNESS, probability=1.0, 
                          params={'factor': 1.2}),
        AugmentationConfig(AugmentationType.FLIP_HORIZONTAL, probability=1.0)
    ]
    
    augmented, applied = augmentor.augment(img, custom_pipeline)
    print(f"    - Applied: {', '.join(applied)}")
    
    # Test dataset augmentation
    print("\n✓ Testing dataset augmentation:")
    output_dir = "test_augmented"
    Path(output_dir).mkdir(exist_ok=True)
    
    results = augmentor.augment_dataset(
        image_paths=[test_image_path],
        output_dir=output_dir,
        augmentations_per_image=3,
        pipeline=augmentor.create_pipeline("minimal")
    )
    
    print(f"    - Generated {len(results)} augmented images")
    print(f"    - Output directory: {output_dir}")
    
    # Statistics
    stats = augmentor.get_statistics()
    print(f"\n✓ Augmentation statistics:")
    print(f"    - Total augmentations: {stats['total_augmentations']}")
    print(f"    - Most used: {stats['most_used']}")
    
    # Cleanup
    Path(test_image_path).unlink()
    import shutil
    shutil.rmtree(output_dir)
    
    print("\n✅ Augmentation test complete")


def test_advanced_augmentation():
    """Test advanced augmentation techniques"""
    print("\n" + "="*70)
    print("TEST 3: ADVANCED AUGMENTATION")
    print("="*70)
    
    # Create two test images
    img1_array = np.ones((256, 256, 3), dtype=np.uint8) * 255
    img1_array[50:200, 50:200] = [200, 100, 100]  # Red
    img1 = Image.fromarray(img1_array)
    
    img2_array = np.ones((256, 256, 3), dtype=np.uint8) * 255
    img2_array[50:200, 50:200] = [100, 100, 200]  # Blue
    img2 = Image.fromarray(img2_array)
    
    augmentor = get_data_augmentor()
    
    # Test MixUp
    print("\n✓ Testing MixUp:")
    mixed = augmentor.mixup(img1, img2, alpha=0.5)
    print(f"    - Blended two images with alpha=0.5")
    print(f"    - Result size: {mixed.size}")
    
    # Test CutMix
    print("\n✓ Testing CutMix:")
    cutmix = augmentor.cutmix(img1, img2, cutout_ratio=0.3)
    print(f"    - Pasted 30% region from image 2 into image 1")
    print(f"    - Result size: {cutmix.size}")
    
    print("\n✅ Advanced augmentation test complete")


def test_incremental_indexing():
    """Test incremental indexing (without actual database)"""
    print("\n" + "="*70)
    print("TEST 4: INCREMENTAL INDEXING")
    print("="*70)
    
    print("\n✓ Incremental indexing features:")
    print("    - Add entries without rebuilding")
    print("    - Update existing entries")
    print("    - Delete entries")
    print("    - Synchronize index")
    print("    - Track changes")
    
    print("\n  Note: Full testing requires database connection")
    print("  See API documentation for usage examples")
    
    # Show example usage
    print("\n  Example usage:")
    print("  ```python")
    print("  indexer = get_incremental_indexer()")
    print("  change = indexer.add_single(")
    print("      file_id=12345,")
    print("      image_path='dataset/images/lab/maple/img_001.jpg',")
    print("      species='maple',")
    print("      source='lab',")
    print("      db=db")
    print("  )")
    print("  print(f'Success: {change.success}')")
    print("  ```")
    
    print("\n✅ Incremental indexing features documented")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("DATA MANAGEMENT FEATURES - TEST SUITE")
    print("="*70)
    
    try:
        # Test 1: Validation
        test_validation()
        
        # Test 2: Augmentation
        test_augmentation()
        
        # Test 3: Advanced augmentation
        test_advanced_augmentation()
        
        # Test 4: Incremental indexing
        test_incremental_indexing()
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("\n✅ All tests completed successfully!")
        print("\nFeatures tested:")
        print("  ✓ Data validation with quality scoring")
        print("  ✓ Image augmentation (minimal, standard, aggressive)")
        print("  ✓ Custom augmentation pipelines")
        print("  ✓ Dataset augmentation")
        print("  ✓ Advanced techniques (MixUp, CutMix)")
        print("  ✓ Incremental indexing (documented)")
        
        print("\nNext steps:")
        print("  1. Review DATA_MANAGEMENT_GUIDE.md for detailed usage")
        print("  2. Test API endpoints with curl or Postman")
        print("  3. Integrate into your data pipeline")
        print("  4. Set up automated validation and augmentation")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

