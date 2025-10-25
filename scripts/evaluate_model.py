"""
Evaluation script for Learning-to-Rank model
This script evaluates the trained model's performance on test data
using various ranking metrics.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import logging
import argparse
from sklearn.metrics import ndcg_score, mean_squared_error, mean_absolute_error
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
    RankingFeatures
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset_split(split_path: str) -> pd.DataFrame:
    """
    Load a dataset split from CSV file
    
    Args:
        split_path: Path to the dataset split CSV file
        
    Returns:
        DataFrame with the dataset split
    """
    logger.info(f"Loading dataset split from {split_path}")
    df = pd.read_csv(split_path)
    logger.info(f"Loaded {len(df)} images from {split_path}")
    return df


def load_trained_model(model_path: str) -> LearningToRankEngine:
    """
    Load a trained Learning-to-Rank model
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded LearningToRankEngine
    """
    logger.info(f"Loading trained model from {model_path}")
    
    # Initialize engine with a default algorithm
    ltr_engine = LearningToRankEngine(algorithm=LTRAlgorithm.LINEAR)
    
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
    
    logger.info("Model loaded successfully")
    return ltr_engine


def create_ranking_features(df: pd.DataFrame) -> Tuple[List[RankingFeatures], List[int]]:
    """
    Create ranking features for the dataset
    In a real scenario, these would come from actual similarity computations
    
    Args:
        df: DataFrame with dataset information
        
    Returns:
        Tuple of (list of ranking features, list of file IDs)
    """
    logger.info(f"Creating ranking features for {len(df)} images...")
    
    features_list = []
    file_ids = []
    
    for idx, row in df.iterrows():
        # Generate realistic feature values based on the image properties
        # In a real implementation, these would come from actual computations
        ranking_features = RankingFeatures(
            vector_similarity=np.random.random(),  # Placeholder - would come from similarity search
            cosine_similarity=np.random.random(),  # Placeholder - would come from similarity search
            euclidean_distance=np.random.random(),  # Placeholder - would come from similarity search
            species_frequency=np.random.random(),  # Placeholder - would come from database stats
            species_popularity=np.random.random(),  # Placeholder - would come from database stats
            image_quality_score=np.random.random(),  # Placeholder - would come from quality assessment
            source_score=1.0 if row['source'] == 'lab' else 0.5,  # Lab=1.0, Field=0.5
            temporal_score=np.random.random(),  # Placeholder - would come from timestamp analysis
            diversity_score=np.random.random(),  # Placeholder - would come from diversity algorithm
            click_through_rate=np.random.random(),  # Placeholder - would come from user feedback
            conversion_rate=np.random.random() # Placeholder - would come from user feedback
        )
        
        features_list.append(ranking_features)
        file_ids.append(row['file_id'])
    
    logger.info(f"Created {len(features_list)} ranking feature sets")
    return features_list, file_ids


def generate_relevance_labels(features_list: List[RankingFeatures]) -> List[float]:
    """
    Generate relevance labels for evaluation data
    In a real scenario, these would come from user feedback or expert annotations
    
    Args:
        features_list: List of ranking features
        
    Returns:
        List of relevance scores (0-5 scale)
    """
    # In a real implementation, relevance scores would come from user feedback
    # For demonstration, we'll generate synthetic relevance scores
    # Higher scores for better features
    relevance_scores = []
    for features in features_list:
        # Create a synthetic relevance score based on feature values
        score = (
            features.vector_similarity * 0.3 +
            features.cosine_similarity * 0.2 +
            (1 - features.euclidean_distance) * 0.2 +  # Inverse since lower distance is better
            features.species_frequency * 0.1 +
            features.image_quality_score * 0.1 +
            features.source_score * 0.1
        )
        # Scale to 0-5 range
        score = min(5.0, max(0.0, score * 5))
        relevance_scores.append(score)
    
    return relevance_scores


def calculate_ranking_metrics(actual_scores: List[float], 
                            predicted_scores: List[float]) -> Dict[str, float]:
    """
    Calculate various ranking metrics
    
    Args:
        actual_scores: Actual relevance scores
        predicted_scores: Predicted scores from the model
        
    Returns:
        Dictionary of evaluation metrics
    """
    actual_array = np.array(actual_scores).reshape(1, -1)
    pred_array = np.array(predicted_scores).reshape(1, -1)
    
    # Mean Squared Error
    mse = mean_squared_error(actual_scores, predicted_scores)
    
    # Mean Absolute Error
    mae = mean_absolute_error(actual_scores, predicted_scores)
    
    # NDCG (Normalized Discounted Cumulative Gain)
    try:
        ndcg = ndcg_score(actual_array, pred_array)
    except:
        ndcg = 0.0  # Fallback if calculation fails
    
    # Calculate Spearman correlation (rank correlation)
    from scipy.stats import spearmanr
    try:
        corr, _ = spearmanr(actual_scores, predicted_scores)
        correlation = corr if not np.isnan(corr) else 0.0
    except:
        correlation = 0.0
    
    return {
        'mse': mse,
        'mae': mae,
        'ndcg': ndcg,
        'correlation': correlation,
        'num_samples': len(actual_scores)
    }


def evaluate_model(test_df: pd.DataFrame, 
                  ltr_engine: LearningToRankEngine) -> Dict[str, float]:
    """
    Evaluate the trained model on test data
    
    Args:
        test_df: Test dataset
        ltr_engine: Trained LearningToRankEngine
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on test set...")
    
    # Create ranking features for test data
    test_features_list, test_file_ids = create_ranking_features(test_df)
    test_relevance_scores = generate_relevance_labels(test_features_list)
    
    # Get predicted scores from the model
    predicted_scores = []
    for features in test_features_list:
        score = ltr_engine.model.score(features)
        predicted_scores.append(score)
    
    # Calculate evaluation metrics
    metrics = calculate_ranking_metrics(test_relevance_scores, predicted_scores)
    
    logger.info(f"Test MSE: {metrics['mse']:.4f}")
    logger.info(f"Test MAE: {metrics['mae']:.4f}")
    logger.info(f"Test NDCG: {metrics['ndcg']:.4f}")
    logger.info(f"Test Correlation: {metrics['correlation']:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Learning-to-Rank model")
    parser.add_argument("--test-data", type=str, default="data/splits/test_dataset.csv",
                       help="Path to test dataset CSV")
    parser.add_argument("--model-path", type=str, default="data/trained_ltr_model.pkl",
                       help="Path to trained model")
    parser.add_argument("--output-file", type=str, default="evaluation_results.txt",
                       help="File to save evaluation results")
    
    args = parser.parse_args()
    
    # Load test dataset
    test_df = load_dataset_split(args.test_data)
    
    # Load trained model
    ltr_engine = load_trained_model(args.model_path)
    
    # Evaluate the model
    metrics = evaluate_model(test_df, ltr_engine)
    
    # Print results
    logger.info("Evaluation complete!")
    logger.info("Final metrics:")
    for key, value in metrics.items():
        logger.info(f" {key}: {value}")
    
    # Save results to file
    with open(args.output_file, 'w') as f:
        f.write("Learning-to-Rank Model Evaluation Results\n")
        f.write("=" * 50 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()