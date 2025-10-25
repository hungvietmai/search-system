"""
Training script for Learning-to-Rank model
This script trains the model using training data, tunes hyperparameters using validation data,
and evaluates performance on test data.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import logging
import argparse
from sklearn.metrics import ndcg_score, mean_squared_error
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
    LinearLTR,
    PairwiseLTR
)
from app.relevance_feedback import get_relevance_feedback_engine

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
    Generate relevance labels for training data
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


def train_ltr_model(train_df: pd.DataFrame,
                   val_df: pd.DataFrame,
                   algorithm: LTRAlgorithm = LTRAlgorithm.LINEAR) -> LearningToRankEngine:
    """
    Train the Learning-to-Rank model using training data and validate on validation data
    
    Args:
        train_df: Training dataset
        val_df: Validation dataset
        algorithm: LTR algorithm to use
        
    Returns:
        Trained LearningToRankEngine
    """
    logger.info("Starting LTR model training...")
    
    # Create ranking features for training data
    train_features_list, train_file_ids = create_ranking_features(train_df)
    train_relevance_scores = generate_relevance_labels(train_features_list)
    
    # Create ranking features for validation data
    val_features_list, val_file_ids = create_ranking_features(val_df)
    val_relevance_scores = generate_relevance_labels(val_features_list)
    
    # Initialize LTR engine
    ltr_engine = LearningToRankEngine(algorithm=algorithm)
    
    # Prepare training data in the format required by the LTR model
    # The model expects a list of feature lists (one per query) and corresponding relevance scores
    training_features_batch = [train_features_list]
    training_relevance_batch = [train_relevance_scores]
    
    # Prepare validation data
    val_features_batch = [val_features_list]
    val_relevance_batch = [val_relevance_scores]
    
    # Train the model with validation monitoring
    if algorithm == LTRAlgorithm.LINEAR or algorithm == LTRAlgorithm.RANKNET:
        # Use the enhanced training method with validation monitoring
        metrics = ltr_engine.train_with_validation(
            train_features=training_features_batch,
            train_relevance=training_relevance_batch,
            val_features=val_features_batch,
            val_relevance=val_relevance_batch,
            learning_rate=0.01,
            iterations=10
        )
    
    # Validate the model on validation set
    logger.info("Validating model on validation set...")
    
    # Calculate validation metrics
    predicted_scores = [ltr_engine.model.score(f) for f in val_features_list]
    actual_scores = val_relevance_scores
    
    mse = mean_squared_error(actual_scores, predicted_scores)
    logger.info(f"Final validation MSE: {mse:.4f}")
    
    return ltr_engine


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
    
    # Calculate test metrics
    predicted_scores = [ltr_engine.model.score(f) for f in test_features_list]
    actual_scores = test_relevance_scores
    
    mse = mean_squared_error(actual_scores, predicted_scores)
    
    # Calculate NDCG (Normalized Discounted Cumulative Gain)
    # Reshape for sklearn (needs 2D array for multiple queries, but we have one "query")
    actual_2d = np.array([actual_scores])
    predicted_2d = np.array([predicted_scores])
    
    try:
        ndcg = ndcg_score(actual_2d, predicted_2d)
    except:
        ndcg = 0.0  # Fallback if calculation fails
    
    metrics = {
        'test_mse': mse,
        'test_ndcg': ndcg,
        'num_test_samples': len(test_df)
    }
    
    logger.info(f"Test MSE: {mse:.4f}")
    logger.info(f"Test NDCG: {ndcg:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Learning-to-Rank model")
    parser.add_argument("--train-data", type=str, default="data/splits/train_dataset.csv",
                       help="Path to training dataset CSV")
    parser.add_argument("--val-data", type=str, default="data/splits/val_dataset.csv",
                       help="Path to validation dataset CSV")
    parser.add_argument("--test-data", type=str, default="data/splits/test_dataset.csv",
                       help="Path to test dataset CSV")
    parser.add_argument("--algorithm", type=str, default="linear",
                       choices=["linear", "ranknet", "lambdamart", "listwise"],
                       help="LTR algorithm to use")
    parser.add_argument("--model-output", type=str, default="data/trained_ltr_model.pkl",
                       help="Path to save trained model")
    
    args = parser.parse_args()
    
    # Convert algorithm string to enum
    algorithm_map = {
        "linear": LTRAlgorithm.LINEAR,
        "ranknet": LTRAlgorithm.RANKNET,
        "lambdamart": LTRAlgorithm.LAMBDAMART,
        "listwise": LTRAlgorithm.LISTWISE
    }
    algorithm = algorithm_map[args.algorithm]
    
    # Load dataset splits
    train_df = load_dataset_split(args.train_data)
    val_df = load_dataset_split(args.val_data)
    test_df = load_dataset_split(args.test_data)
    
    # Train the model
    ltr_engine = train_ltr_model(
        train_df,
        val_df,
        algorithm=algorithm
    )
    
    # Evaluate the model
    metrics = evaluate_model(
        test_df,
        ltr_engine
    )
    
    # Save the trained model
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model with algorithm info for proper loading
    save_data = {
        'algorithm': algorithm.value,
        'weights': ltr_engine.model.weights if hasattr(ltr_engine.model, 'weights') else None
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    logger.info(f"Trained model saved to {args.model_output}")
    
    # Print final metrics
    logger.info("Training complete!")
    logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()