"""
Learning-to-Rank (LTR) Module
Combines multiple similarity scores using machine learning algorithms

Supported algorithms:
- RankNet: Pairwise ranking
- LambdaMART: Gradient boosted decision trees
- Linear combination: Weighted linear model
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LTRAlgorithm(Enum):
    """Learning-to-Rank algorithms"""
    LINEAR = "linear"           # Weighted linear combination
    RANKNET = "ranknet"         # Neural pairwise ranking
    LAMBDAMART = "lambdamart"   # Gradient boosted trees
    LISTWISE = "listwise"       # Listwise ranking


@dataclass
class RankingFeatures:
    """Features for learning-to-rank"""
    # Similarity scores
    vector_similarity: float
    cosine_similarity: float
    euclidean_distance: float
    
    # Frequency-based features
    species_frequency: float
    species_popularity: float
    
    # Quality features
    image_quality_score: float
    source_score: float  # lab=1.0, field=0.5
    
    # Context features
    temporal_score: float  # Recent images scored higher
    diversity_score: float
    
    # Interaction features
    click_through_rate: float  # Historical CTR
    conversion_rate: float     # How often users select this
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models"""
        return np.array([
            self.vector_similarity,
            self.cosine_similarity,
            self.euclidean_distance,
            self.species_frequency,
            self.species_popularity,
            self.image_quality_score,
            self.source_score,
            self.temporal_score,
            self.diversity_score,
            self.click_through_rate,
            self.conversion_rate
        ])
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names"""
        return [
            'vector_similarity',
            'cosine_similarity',
            'euclidean_distance',
            'species_frequency',
            'species_popularity',
            'image_quality_score',
            'source_score',
            'temporal_score',
            'diversity_score',
            'click_through_rate',
            'conversion_rate'
        ]


class LinearLTR:
    """
    Linear Learning-to-Rank
    Simple weighted combination of features
    """
    
    def __init__(self, weights: Optional[np.ndarray] = None):
        """
        Initialize linear LTR
        
        Args:
            weights: Feature weights (11 features)
        """
        if weights is None:
            # Default weights (can be learned from data)
            self.weights = np.array([
                0.30,  # vector_similarity
                0.20,  # cosine_similarity
                -0.10, # euclidean_distance (negative because smaller is better)
                0.10,  # species_frequency
                0.05,  # species_popularity
                0.10,  # image_quality_score
                0.05,  # source_score
                0.05,  # temporal_score
                0.05,  # diversity_score
                0.05,  # click_through_rate
                0.05   # conversion_rate
            ])
        else:
            self.weights = weights
    
    def score(self, features: RankingFeatures) -> float:
        """
        Compute ranking score
        
        Args:
            features: Ranking features
            
        Returns:
            Ranking score
        """
        feature_array = features.to_array()
        score = np.dot(self.weights, feature_array)
        return float(score)
    
    def rank(self, features_list: List[RankingFeatures]) -> List[int]:
        """
        Rank items by score
        
        Args:
            features_list: List of features for each item
            
        Returns:
            Ranked indices (descending order)
        """
        scores = [self.score(f) for f in features_list]
        ranked_indices = np.argsort(scores)[::-1]  # Descending
        return ranked_indices.tolist()
    
    def learn_weights(self, 
                     features_list: List[List[RankingFeatures]],
                     relevance_scores: List[List[float]],
                     learning_rate: float = 0.01,
                     iterations: int = 100):
        """
        Learn weights from training data
        
        Args:
            features_list: List of feature lists (one per query)
            relevance_scores: List of relevance score lists (one per query)
            learning_rate: Learning rate for gradient descent
            iterations: Number of iterations
        """
        logger.info("Learning linear LTR weights...")
        
        for iteration in range(iterations):
            total_loss = 0.0
            gradient = np.zeros_like(self.weights)
            
            # For each query
            for features, relevances in zip(features_list, relevance_scores):
                # Convert to arrays
                X = np.array([f.to_array() for f in features])
                y = np.array(relevances)
                
                # Predict scores
                predictions = X @ self.weights
                
                # Compute loss (MSE)
                loss = np.mean((predictions - y) ** 2)
                total_loss += loss
                
                # Compute gradient
                grad = 2 * X.T @ (predictions - y) / len(y)
                gradient += grad
            
            # Update weights
            gradient /= len(features_list)
            self.weights -= learning_rate * gradient
            
            if (iteration + 1) % 10 == 0:
                avg_loss = total_loss / len(features_list)
                logger.info(f"Iteration {iteration + 1}/{iterations}, Loss: {avg_loss:.4f}")
        
        logger.info(f"Learned weights: {self.weights}")
    
    def save(self, path: Path):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)
        logger.info(f"Saved linear LTR model to {path}")
    
    def load(self, path: Path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            self.weights = pickle.load(f)
        logger.info(f"Loaded linear LTR model from {path}")


class PairwiseLTR:
    """
    Pairwise Learning-to-Rank (RankNet-style)
    Learns to predict relative ordering of pairs
    """
    
    def __init__(self, feature_dim: int = 11):
        """
        Initialize pairwise LTR
        
        Args:
            feature_dim: Number of features
        """
        self.feature_dim = feature_dim
        # Simple linear model (can be extended to neural network)
        self.weights = np.random.randn(feature_dim) * 0.01
    
    def score(self, features: RankingFeatures) -> float:
        """Compute ranking score"""
        feature_array = features.to_array()
        score = np.dot(self.weights, feature_array)
        return float(score)
    
    def sigmoid(self, x: float) -> float:
        """Sigmoid activation"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def pairwise_loss(self, score_i: float, score_j: float, 
                      rel_i: float, rel_j: float) -> float:
        """
        Compute pairwise loss
        
        Args:
            score_i: Score of item i
            score_j: Score of item j
            rel_i: Relevance of item i
            rel_j: Relevance of item j
            
        Returns:
            Loss value
        """
        # If i is more relevant than j, score_i should be higher
        if rel_i > rel_j:
            # Probability that i is ranked higher than j
            prob = self.sigmoid(score_i - score_j)
            # Cross-entropy loss
            loss = -np.log(prob + 1e-8)
        elif rel_i < rel_j:
            prob = self.sigmoid(score_j - score_i)
            loss = -np.log(prob + 1e-8)
        else:
            # Equal relevance, no loss
            loss = 0.0
        
        return loss
    
    def train(self,
             features_list: List[List[RankingFeatures]],
             relevance_scores: List[List[float]],
             learning_rate: float = 0.01,
             iterations: int = 100):
        """
        Train pairwise ranking model
        
        Args:
            features_list: List of feature lists (one per query)
            relevance_scores: List of relevance score lists
            learning_rate: Learning rate
            iterations: Number of iterations
        """
        logger.info("Training pairwise LTR model...")
        
        for iteration in range(iterations):
            total_loss = 0.0
            gradient = np.zeros_like(self.weights)
            
            # For each query
            for features, relevances in zip(features_list, relevance_scores):
                # Generate all pairs
                n = len(features)
                for i in range(n):
                    for j in range(i + 1, n):
                        # Compute scores
                        score_i = self.score(features[i])
                        score_j = self.score(features[j])
                        
                        # Compute loss
                        loss = self.pairwise_loss(
                            score_i, score_j,
                            relevances[i], relevances[j]
                        )
                        total_loss += loss
                        
                        # Compute gradient
                        if relevances[i] > relevances[j]:
                            # i should be ranked higher
                            prob = self.sigmoid(score_i - score_j)
                            grad = -(1 - prob) * (features[i].to_array() - features[j].to_array())
                        elif relevances[i] < relevances[j]:
                            # j should be ranked higher
                            prob = self.sigmoid(score_j - score_i)
                            grad = (1 - prob) * (features[i].to_array() - features[j].to_array())
                        else:
                            grad = np.zeros_like(self.weights)
                        
                        gradient += grad
            
            # Update weights
            if len(features_list) > 0:
                gradient /= sum(len(f) * (len(f) - 1) / 2 for f in features_list)
                self.weights -= learning_rate * gradient
            
            if (iteration + 1) % 10 == 0:
                avg_loss = total_loss / max(1, sum(len(f) * (len(f) - 1) / 2 for f in features_list))
                logger.info(f"Iteration {iteration + 1}/{iterations}, Loss: {avg_loss:.4f}")
        
        logger.info("Pairwise LTR training complete")
    
    def rank(self, features_list: List[RankingFeatures]) -> List[int]:
        """Rank items by score"""
        scores = [self.score(f) for f in features_list]
        ranked_indices = np.argsort(scores)[::-1]
        return ranked_indices.tolist()


class LearningToRankEngine:
    """
    Main Learning-to-Rank Engine
    Supports multiple algorithms and online learning
    """
    
    def __init__(self, 
                 algorithm: LTRAlgorithm = LTRAlgorithm.LINEAR,
                 model_path: Optional[Path] = None):
        """
        Initialize LTR engine
        
        Args:
            algorithm: LTR algorithm to use
            model_path: Path to saved model
        """
        self.algorithm = algorithm
        self.model_path = model_path or Path("data/ltr_model.pkl")
        
        # Initialize model
        if algorithm == LTRAlgorithm.LINEAR:
            self.model = LinearLTR()
        elif algorithm == LTRAlgorithm.RANKNET:
            self.model = PairwiseLTR()
        else:
            # Default to linear
            logger.warning(f"Algorithm {algorithm} not fully implemented, using LINEAR")
            self.model = LinearLTR()
        
        # Load model if exists
        if self.model_path.exists() and hasattr(self.model, 'load'):
            try:
                self.model.load(self.model_path)  # type: ignore
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
    
    def rank(self, 
            candidates: List[Dict],
            features_list: List[RankingFeatures]) -> List[Dict]:
        """
        Rank candidates using LTR model
        
        Args:
            candidates: List of candidate items
            features_list: List of features for each candidate
            
        Returns:
            Ranked candidates
        """
        if len(candidates) != len(features_list):
            raise ValueError("Number of candidates must match number of features")
        
        # Get ranking indices
        ranked_indices = self.model.rank(features_list)
        
        # Reorder candidates
        ranked_candidates = [candidates[i] for i in ranked_indices]
        
        return ranked_candidates
    
    def update_model(self,
                    features_list: List[List[RankingFeatures]],
                    relevance_scores: List[List[float]]):
        """
        Update model with new training data
        
        Args:
            features_list: List of feature lists (queries)
            relevance_scores: List of relevance scores (queries)
        """
        if hasattr(self.model, 'learn_weights'):
            self.model.learn_weights(features_list, relevance_scores)  # type: ignore
        elif hasattr(self.model, 'train'):
            self.model.train(features_list, relevance_scores)  # type: ignore
        
        # Save updated model
        self.save_model()
    
    def save_model(self):
        """Save model to disk"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, 'save'):
            self.model.save(self.model_path)  # type: ignore


# Global LTR engine instance
_ltr_engine = None


def get_ltr_engine(algorithm: LTRAlgorithm = LTRAlgorithm.LINEAR) -> LearningToRankEngine:
    """
    Get or create global LTR engine
    
    Args:
        algorithm: LTR algorithm to use
        
    Returns:
        LearningToRankEngine instance
    """
    global _ltr_engine
    
    if _ltr_engine is None:
        _ltr_engine = LearningToRankEngine(algorithm=algorithm)
    
    return _ltr_engine

