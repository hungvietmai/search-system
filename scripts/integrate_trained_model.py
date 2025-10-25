"""
Script to demonstrate how to integrate the trained Learning-to-Rank model into search functionality
This shows how to use the trained model to improve search results
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict
import logging
import pickle

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from app.database import get_db_context
from app.models import LeafImage
from app.feature_extractor import get_feature_extractor
from app.learning_to_rank import (
    LearningToRankEngine, 
    LTRAlgorithm, 
    RankingFeatures,
    get_ltr_engine
)
from app.faiss_client import get_faiss_client
from app.similarity_metrics import MetricCalculator, SimilarityMetric

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_trained_model(model_path: str = "data/trained_ltr_model.pkl") -> LearningToRankEngine:
    """
    Load the trained Learning-to-Rank model
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        Loaded LearningToRankEngine
    """
    logger.info(f"Loading trained model from {model_path}")
    
    # Initialize engine with a default algorithm
    ltr_engine = get_ltr_engine(algorithm=LTRAlgorithm.LINEAR)
    
    # Load the model weights
    if Path(model_path).exists():
        # We need to manually load the weights since the save/load methods are model-specific
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        # Determine model type and restore weights
        if isinstance(saved_data, dict) and 'weights' in saved_data:
            if saved_data['algorithm'] == 'linear':
                from app.learning_to_rank import LinearLTR
                ltr_engine.model = LinearLTR(weights=saved_data['weights'])
                ltr_engine.algorithm = LTRAlgorithm.LINEAR
            elif saved_data['algorithm'] == 'ranknet':
                from app.learning_to_rank import PairwiseLTR
                ltr_engine.model = PairwiseLTR()
                ltr_engine.model.weights = saved_data['weights']
                ltr_engine.algorithm = LTRAlgorithm.RANKNET
        else:
            # For backward compatibility with simple weights
            ltr_engine.model.weights = saved_data
    
    logger.info("Trained model loaded successfully")
    return ltr_engine


def create_search_ranking_features(query_features: np.ndarray, 
                                  candidate_features: List[np.ndarray],
                                  candidate_info: List[Dict]) -> List[RankingFeatures]:
    """
    Create ranking features for search results
    
    Args:
        query_features: Features of the query image
        candidate_features: Features of candidate images
        candidate_info: Information about candidates (species, source, etc.)
        
    Returns:
        List of RankingFeatures for each candidate
    """
    features_list = []
    
    for i, (candidate_feat, info) in enumerate(zip(candidate_features, candidate_info)):
        # Calculate similarity metrics between query and candidate
        vector_similarity = MetricCalculator.cosine_similarity(query_features, candidate_feat)
        cosine_similarity = MetricCalculator.cosine_similarity(query_features, candidate_feat)
        euclidean_distance = MetricCalculator.l2_distance(query_features, candidate_feat)
        
        # Create ranking features based on available information
        ranking_features = RankingFeatures(
            vector_similarity=float(vector_similarity),
            cosine_similarity=float(cosine_similarity),
            euclidean_distance=float(euclidean_distance),
            species_frequency=info.get('species_frequency', 0.5),  # Would come from database
            species_popularity=info.get('species_popularity', 0.5),  # Would come from database
            image_quality_score=info.get('image_quality_score', 0.8),  # Would come from quality assessment
            source_score=1.0 if info.get('source') == 'lab' else 0.5,  # Lab=1.0, Field=0.5
            temporal_score=info.get('temporal_score', 0.5),  # Would come from timestamp analysis
            diversity_score=info.get('diversity_score', 0.5),  # Would come from diversity algorithm
            click_through_rate=info.get('click_through_rate', 0.1),  # Would come from user feedback
            conversion_rate=info.get('conversion_rate', 0.1)  # Would come from user feedback
        )
        
        features_list.append(ranking_features)
    
    return features_list


def improved_search_with_ltr(query_image_path: str, 
                            top_k: int = 10,
                            model_path: str = "data/trained_ltr_model.pkl") -> List[Dict]:
    """
    Perform search using the trained Learning-to-Rank model to improve results
    
    Args:
        query_image_path: Path to the query image
        top_k: Number of results to return
        model_path: Path to the trained model
        
    Returns:
        List of improved search results
    """
    logger.info(f"Performing improved search for {query_image_path}")
    
    # Load the trained model
    ltr_engine = load_trained_model(model_path)
    
    # Initialize feature extractor and Faiss client
    feature_extractor = get_feature_extractor()
    faiss_client = get_faiss_client()
    
    # Extract features for the query image
    query_features = feature_extractor.extract_features(query_image_path, is_query=True)
    
    # Perform initial similarity search using Faiss
    logger.info("Performing initial similarity search...")
    initial_results = faiss_client.search(query_features, top_k=top_k*2)  # Get more results for re-ranking
    
    # Get detailed information about candidates from database
    candidate_ids = [result['file_id'] for result in initial_results]
    
    with get_db_context() as db:
        # Get candidate details from database
        candidates = []
        for file_id in candidate_ids:
            leaf_img = db.query(LeafImage).filter(LeafImage.file_id == file_id).first()
            if leaf_img:
                candidates.append({
                    'file_id': leaf_img.file_id,
                    'image_path': leaf_img.image_path,
                    'species': leaf_img.species,
                    'source': leaf_img.source
                })
    
    # Get features for all candidates
    candidate_features = []
    for result in initial_results:
        # In a real implementation, we'd get these from the index
        # For now, we'll just use placeholder values
        candidate_features.append(np.random.rand(len(query_features)))  # Placeholder
    
    # Create ranking features for all candidates
    ranking_features_list = create_search_ranking_features(query_features, candidate_features, candidates)
    
    # Use the trained LTR model to re-rank the results
    logger.info("Re-ranking results with trained LTR model...")
    re_ranked_candidates = ltr_engine.rank(candidates, ranking_features_list)
    
    # Return only the top-k results after re-ranking
    final_results = re_ranked_candidates[:top_k]
    
    logger.info(f"Search completed, returned {len(final_results)} results")
    return final_results


def demonstrate_integration():
    """
    Demonstrate how to use the trained model in search functionality
    """
    logger.info("Demonstrating integration of trained model into search...")
    
    # Example of how to use the improved search function
    # Note: You would replace 'path/to/query/image.jpg' with an actual image path
    print("""
    Example usage in your search application:

    # In your search endpoint (e.g., in app/main.py)
    from scripts.integrate_trained_model import improved_search_with_ltr

    @app.post("/search-improved", response_model=SearchResponse)
    async def improved_search_similar_leaves(
        file: UploadFile = File(...),
        top_k: int = Query(10, ge=1, le=100),
        db: Session = Depends(get_db)
    ):
        # Save uploaded file temporarily
        temp_path = f"temp/{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        try:
            # Use the improved search with LTR
            results = improved_search_with_ltr(
                query_image_path=temp_path,
                top_k=top_k,
                model_path="data/trained_ltr_model.pkl"
            )
            
            return SearchResponse(results=results, query_image=file.filename)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    """)
    
    # Example of how to periodically retrain the model with user feedback
    print("""
    Example of how to periodically retrain with user feedback:

    # In a background job or scheduled task
    def retrain_model_with_feedback():
        # Collect user feedback data
        feedback_data = collect_user_feedback()
        
        # Split into train/validation/test
        train_data, val_data, test_data = split_feedback_data(feedback_data)
        
        # Retrain the model
        ltr_engine = train_ltr_model(train_data, val_data)
        
        # Evaluate and save if performance improved
        metrics = evaluate_model(test_data, ltr_engine)
        if metrics['ndcg'] > previous_best_ndcg:
            ltr_engine.save_model()  # Save if performance improved
    """)


def main():
    print("Learning-to-Rank Integration Guide")
    print("=" * 50)
    
    # Show the integration examples
    demonstrate_integration()
    
    print("\n" + "=" * 50)
    print("Integration Steps:")
    print("1. Train your model using: python scripts/train_ltr_model.py")
    print("2. Use the trained model in search by calling improved_search_with_ltr()")
    print("3. Integrate the improved search into your API endpoints")
    print("4. Optionally, retrain periodically with user feedback")


if __name__ == "__main__":
    main()