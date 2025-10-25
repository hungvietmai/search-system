"""
Test script to verify that the trained LTR model can be loaded properly
"""
import pickle
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.learning_to_rank import get_ltr_engine, LTRAlgorithm, LinearLTR, PairwiseLTR

def test_model_loading():
    """
    Test loading the trained model
    """
    model_path = Path("data/trained_ltr_model.pkl")
    
    print(f"Testing model loading from {model_path}")
    print(f"Model file exists: {model_path.exists()}")
    
    if not model_path.exists():
        print("❌ Model file does not exist!")
        return False
    
    print(f"Model file size: {model_path.stat().st_size} bytes")
    
    try:
        # Load the saved data
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        print(f"✓ Model file loaded successfully")
        print(f"Saved data type: {type(saved_data)}")
        print(f"Saved data content: {saved_data}")
        
        # Initialize LTR engine
        ltr_engine = get_ltr_engine(algorithm=LTRAlgorithm.LINEAR)
        
        # Determine model type and restore weights
        if isinstance(saved_data, dict) and 'weights' in saved_data:
            if saved_data['algorithm'] == 'linear':
                print("✓ Loading LinearLTR model")
                ltr_engine.model = LinearLTR(weights=saved_data['weights'])
                ltr_engine.algorithm = LTRAlgorithm.LINEAR
            elif saved_data['algorithm'] == 'ranknet':
                print("✓ Loading PairwiseLTR model")
                ltr_engine.model = PairwiseLTR()
                ltr_engine.model.weights = saved_data['weights']
                ltr_engine.algorithm = LTRAlgorithm.RANKNET
        else:
            # For backward compatibility with simple weights
            print("✓ Loading model with simple weights")
            ltr_engine.model.weights = saved_data
        
        print(f"✓ Model loaded into LTR engine successfully")
        print(f"Model weights shape: {ltr_engine.model.weights.shape if hasattr(ltr_engine.model, 'weights') else 'N/A'}")
        
        # Test creating a simple ranking feature and scoring
        from app.learning_to_rank import RankingFeatures
        import numpy as np
        
        # Create a sample ranking feature
        sample_features = RankingFeatures(
            vector_similarity=0.8,
            cosine_similarity=0.8,
            euclidean_distance=0.5,
            species_frequency=0.6,
            species_popularity=0.7,
            image_quality_score=0.9,
            source_score=1.0,  # lab
            temporal_score=0.5,
            diversity_score=0.4,
            click_through_rate=0.3,
            conversion_rate=0.2
        )
        
        # Test scoring
        score = ltr_engine.model.score(sample_features)
        print(f"✓ Sample feature scored successfully: {score}")
        
        # Test ranking
        features_list = [sample_features] * 5  # Create 5 identical features
        ranked_indices = ltr_engine.model.rank(features_list)
        print(f"✓ Ranking test successful: {ranked_indices}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Trained LTR Model Loading")
    print("="*40)
    
    success = test_model_loading()
    
    print()
    if success:
        print("✓ All model loading tests passed!")
    else:
        print("✗ Model loading tests failed!")