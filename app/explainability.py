"""
Search Result Explainability and Confidence Scoring
Makes search results transparent and trustworthy

Features:
- Result explanations (why this match?)
- Confidence scoring
- Feature contribution analysis
- Visual similarity explanation
- Species-level insights
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from sqlalchemy.orm import Session

from app.models import LeafImage

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level categories"""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"            # 75-90%
    MEDIUM = "medium"        # 50-75%
    LOW = "low"              # 25-50%
    VERY_LOW = "very_low"    # 0-25%


@dataclass
class ExplanationComponent:
    """Single explanation component"""
    factor: str
    score: float
    weight: float
    description: str
    contribution: float  # How much this factor contributed to the match


@dataclass
class SearchExplanation:
    """Complete explanation for a search result"""
    file_id: int
    species: str
    confidence_score: float
    confidence_level: ConfidenceLevel
    distance: float
    overall_explanation: str
    components: List[ExplanationComponent]
    visual_similarities: List[str]
    potential_concerns: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'file_id': self.file_id,
            'species': self.species,
            'confidence_score': round(self.confidence_score, 4),
            'confidence_level': self.confidence_level.value,
            'distance': round(self.distance, 4),
            'overall_explanation': self.overall_explanation,
            'components': [
                {
                    'factor': c.factor,
                    'score': round(c.score, 4),
                    'weight': round(c.weight, 4),
                    'description': c.description,
                    'contribution': round(c.contribution, 4)
                }
                for c in self.components
            ],
            'visual_similarities': self.visual_similarities,
            'potential_concerns': self.potential_concerns
        }


class ConfidenceScorer:
    """Calculate confidence scores for search results"""
    
    def __init__(self):
        """Initialize confidence scorer"""
        # Thresholds for distance (L2)
        self.distance_thresholds = {
            'excellent': 0.4,   # More permissive threshold
            'good': 0.7,        # More permissive threshold
            'fair': 1.1,        # More permissive threshold
            'poor': 1.6         # More permissive threshold
        }
    
    def distance_to_similarity(self, distance: float) -> float:
        """
        Convert L2 distance to similarity score (0-1)
        
        Args:
            distance: L2 distance
            
        Returns:
            Similarity score (higher is better)
        """
        # Use exponential decay
        # similarity = exp(-k * distance)
        k = 1.0  # Decay rate
        similarity = np.exp(-k * distance)
        return float(similarity)
    
    def calculate_distance_score(self, distance: float) -> Tuple[float, str]:
        """
        Calculate score based on distance
        
        Args:
            distance: L2 distance
            
        Returns:
            Tuple of (score, description)
        """
        similarity = self.distance_to_similarity(distance)
        
        if distance < self.distance_thresholds['excellent']:
            desc = "Extremely similar visual features"
        elif distance < self.distance_thresholds['good']:
            desc = "Very similar visual features"
        elif distance < self.distance_thresholds['fair']:
            desc = "Moderately similar visual features"
        elif distance < self.distance_thresholds['poor']:
            desc = "Somewhat similar visual features"
        else:
            desc = "Limited visual similarity"
        
        return similarity, desc
    
    def calculate_rank_score(self, position: int, total_results: int) -> Tuple[float, str]:
        """
        Calculate score based on rank position
        
        Args:
            position: Position in results (0-based)
            total_results: Total number of results
            
        Returns:
            Tuple of (score, description)
        """
        # Higher rank = higher score
        rank_score = 1.0 - (position / max(total_results, 10))
        
        if position == 0:
            desc = "Top match among all results"
        elif position < 3:
            desc = f"Ranked #{position + 1}, high confidence"
        elif position < 5:
            desc = f"Ranked #{position + 1}, good match"
        else:
            desc = f"Ranked #{position + 1}"
        
        return rank_score, desc
    
    def calculate_species_frequency_score(self,
                                         species: str,
                                         db: Session) -> Tuple[float, str]:
        """
        Calculate score based on species frequency in database
        
        Args:
            species: Species name
            db: Database session
            
        Returns:
            Tuple of (score, description)
        """
        # Count images for this species
        count = db.query(LeafImage).filter(LeafImage.species == species).count()
        
        # More images = higher confidence (better training data)
        if count > 100:
            score = 1.0
            desc = f"Well-represented species ({count}+ reference images)"
        elif count > 50:
            score = 0.8
            desc = f"Common species ({count} reference images)"
        elif count > 20:
            score = 0.6
            desc = f"Moderately represented ({count} reference images)"
        elif count > 10:
            score = 0.4
            desc = f"Limited reference data ({count} images)"
        else:
            score = 0.2
            desc = f"Rare species with limited data ({count} images)"
        
        return score, desc
    
    def calculate_source_score(self, source: str) -> Tuple[float, str]:
        """
        Calculate score based on image source
        
        Args:
            source: Image source ('lab' or 'field')
            
        Returns:
            Tuple of (score, description)
        """
        if source == 'lab':
            score = 1.0
            desc = "High-quality lab photograph (controlled conditions)"
        else:
            score = 0.7
            desc = "Field photograph (natural conditions)"
        
        return score, desc
    
    def calculate_confidence(self,
                           distance: float,
                           position: int,
                           total_results: int,
                           species: str,
                           source: str,
                           db: Session) -> Tuple[float, List[ExplanationComponent]]:
        """
        Calculate overall confidence score
        
        Args:
            distance: L2 distance
            position: Rank position
            total_results: Total results
            species: Species name
            source: Image source
            db: Database session
            
        Returns:
            Tuple of (confidence_score, explanation_components)
        """
        components = []
        
        # 1. Distance-based score (50% weight) - Increased to prioritize visual similarity
        dist_score, dist_desc = self.calculate_distance_score(distance)
        components.append(ExplanationComponent(
            factor="Visual Similarity",
            score=dist_score,
            weight=0.50,
            description=dist_desc,
            contribution=dist_score * 0.50
        ))
        
        # 2. Rank-based score (20% weight)
        rank_score, rank_desc = self.calculate_rank_score(position, total_results)
        components.append(ExplanationComponent(
            factor="Search Ranking",
            score=rank_score,
            weight=0.20,
            description=rank_desc,
            contribution=rank_score * 0.20
        ))
        
        # 3. Species frequency score (20% weight) - Reduced slightly
        freq_score, freq_desc = self.calculate_species_frequency_score(species, db)
        components.append(ExplanationComponent(
            factor="Species Data Quality",
            score=freq_score,
            weight=0.20,
            description=freq_desc,
            contribution=freq_score * 0.20
        ))
        
        # 4. Source quality score (10% weight) - Reduced to prioritize visual similarity
        source_score, source_desc = self.calculate_source_score(source)
        components.append(ExplanationComponent(
            factor="Image Quality",
            score=source_score,
            weight=0.10,
            description=source_desc,
            contribution=source_score * 0.10
        ))
        
        # Calculate weighted confidence score
        confidence = sum(c.contribution for c in components)
        
        return confidence, components
    
    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """
        Get confidence level category
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            ConfidenceLevel
        """
        if confidence >= 0.90:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.50:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class ResultExplainer:
    """Generate explanations for search results"""
    
    def __init__(self):
        """Initialize result explainer"""
        self.confidence_scorer = ConfidenceScorer()
    
    def analyze_visual_features(self,
                               query_features: np.ndarray,
                               result_features: np.ndarray) -> List[str]:
        """
        Analyze which visual features are similar
        
        Args:
            query_features: Query feature vector
            result_features: Result feature vector
            
        Returns:
            List of visual similarity descriptions
        """
        similarities = []
        
        # Compute feature-wise similarity
        # Split into regions (rough approximation)
        n_features = len(query_features)
        region_size = n_features // 4
        
        # Analyze different regions
        regions = {
            'overall_shape': (0, region_size),
            'texture_patterns': (region_size, 2 * region_size),
            'edge_characteristics': (2 * region_size, 3 * region_size),
            'color_distribution': (3 * region_size, n_features)
        }
        
        for region_name, (start, end) in regions.items():
            query_region = query_features[start:end]
            result_region = result_features[start:end]
            
            # Compute cosine similarity for this region
            dot_product = np.dot(query_region, result_region)
            norm_product = np.linalg.norm(query_region) * np.linalg.norm(result_region)
            
            if norm_product > 0:
                similarity = dot_product / norm_product
                
                if similarity > 0.85:
                    similarities.append(f"Very similar {region_name.replace('_', ' ')}")
                elif similarity > 0.70:
                    similarities.append(f"Similar {region_name.replace('_', ' ')}")
        
        return similarities if similarities else ["General visual similarity"]
    
    def identify_concerns(self,
                         confidence: float,
                         distance: float,
                         position: int,
                         species_count: int) -> List[str]:
        """
        Identify potential concerns about the match
        
        Args:
            confidence: Confidence score
            distance: L2 distance
            position: Rank position
            species_count: Number of images for this species
            
        Returns:
            List of concerns
        """
        concerns = []
        
        if confidence < 0.50:
            concerns.append("Low overall confidence - consider this a tentative match")
        
        if distance > 1.0:
            concerns.append("Substantial visual differences detected")
        
        if position > 5:
            concerns.append("Not among top matches - alternative species may be more likely")
        
        if species_count < 20:
            concerns.append("Limited reference data for this species - results may be less reliable")
        
        return concerns
    
    def generate_overall_explanation(self,
                                    confidence: float,
                                    confidence_level: ConfidenceLevel,
                                    species: str,
                                    distance: float) -> str:
        """
        Generate overall explanation text
        
        Args:
            confidence: Confidence score
            confidence_level: Confidence level
            species: Species name
            distance: Distance score
            
        Returns:
            Explanation text
        """
        if confidence_level == ConfidenceLevel.VERY_HIGH:
            return (f"Very high confidence match for {species}. The visual features show "
                   f"strong similarity with minimal differences (distance: {distance:.3f}).")
        
        elif confidence_level == ConfidenceLevel.HIGH:
            return (f"High confidence match for {species}. Visual analysis indicates "
                   f"substantial similarity (distance: {distance:.3f}).")
        
        elif confidence_level == ConfidenceLevel.MEDIUM:
            return (f"Moderate confidence match for {species}. Some visual similarities "
                   f"present but differences noted (distance: {distance:.3f}).")
        
        elif confidence_level == ConfidenceLevel.LOW:
            return (f"Low confidence match for {species}. Limited visual similarity "
                   f"detected (distance: {distance:.3f}). Consider as a tentative match.")
        
        else:
            return (f"Very low confidence match for {species}. Substantial visual differences "
                   f"(distance: {distance:.3f}). This may not be the correct species.")
    
    def explain_result(self,
                      file_id: int,
                      species: str,
                      source: str,
                      distance: float,
                      position: int,
                      total_results: int,
                      query_features: Optional[np.ndarray],
                      result_features: Optional[np.ndarray],
                      db: Session) -> SearchExplanation:
        """
        Generate complete explanation for a search result
        
        Args:
            file_id: File ID
            species: Species name
            source: Image source
            distance: L2 distance
            position: Rank position (0-based)
            total_results: Total number of results
            query_features: Query feature vector (optional)
            result_features: Result feature vector (optional)
            db: Database session
            
        Returns:
            SearchExplanation
        """
        # Calculate confidence
        confidence, components = self.confidence_scorer.calculate_confidence(
            distance, position, total_results, species, source, db
        )
        
        # Get confidence level
        confidence_level = self.confidence_scorer.get_confidence_level(confidence)
        
        # Analyze visual features if available
        if query_features is not None and result_features is not None:
            visual_similarities = self.analyze_visual_features(
                query_features, result_features
            )
        else:
            visual_similarities = [
                "Visual feature analysis unavailable",
                f"Match based on distance score: {distance:.3f}"
            ]
        
        # Get species count
        species_count = db.query(LeafImage).filter(
            LeafImage.species == species
        ).count()
        
        # Identify concerns
        concerns = self.identify_concerns(
            confidence, distance, position, species_count
        )
        
        # Generate overall explanation
        overall_explanation = self.generate_overall_explanation(
            confidence, confidence_level, species, distance
        )
        
        return SearchExplanation(
            file_id=file_id,
            species=species,
            confidence_score=confidence,
            confidence_level=confidence_level,
            distance=distance,
            overall_explanation=overall_explanation,
            components=components,
            visual_similarities=visual_similarities,
            potential_concerns=concerns
        )
    
    def explain_top_results(self,
                           results: List[Dict],
                           query_features: Optional[np.ndarray],
                           db: Session) -> List[SearchExplanation]:
        """
        Generate explanations for all results
        
        Args:
            results: List of search results
            query_features: Query feature vector
            db: Database session
            
        Returns:
            List of explanations
        """
        explanations = []
        
        # Import feature cache and feature extractor to get result features
        from app.cache import get_feature_cache
        from app.feature_extractor import get_feature_extractor
        feature_cache = get_feature_cache()
        feature_extractor = get_feature_extractor()
        
        for i, result in enumerate(results):
            # Try to get result features from cache first
            result_features = feature_cache.get_features(result['file_id'])
            
            # If not in cache, try to extract features from the image
            if result_features is None:
                try:
                    # Get image path from database using file_id
                    image_record = db.query(LeafImage).filter(LeafImage.file_id == result['file_id']).first()
                    if image_record and image_record.image_path:
                        # Construct absolute path from settings.dataset_path
                        from pathlib import Path
                        from config import settings
                        abs_path = Path(settings.dataset_path).parent / image_record.image_path
                        # Extract features for the result image
                        result_features = feature_extractor.extract_features(str(abs_path), is_query=False)
                        # Cache the features for future use
                        feature_cache.set_features(result['file_id'], result_features)
                except Exception as e:
                    logger.warning(f"Failed to extract features for file_id {result['file_id']}: {e}")
            
            explanation = self.explain_result(
                file_id=result['file_id'],
                species=result['species'],
                source=result['source'],
                distance=result['distance'],
                position=i,
                total_results=len(results),
                query_features=query_features,
                result_features=result_features,
                db=db
            )
            explanations.append(explanation)
        
        return explanations
    
    def compare_results(self,
                       result1: SearchExplanation,
                       result2: SearchExplanation) -> str:
        """
        Compare two results and explain the difference
        
        Args:
            result1: First result
            result2: Second result
            
        Returns:
            Comparison explanation
        """
        conf_diff = result1.confidence_score - result2.confidence_score
        
        if abs(conf_diff) < 0.05:
            return (f"Both {result1.species} and {result2.species} are equally likely "
                   f"matches with similar confidence scores.")
        
        elif conf_diff > 0:
            return (f"{result1.species} is more likely than {result2.species} "
                   f"(confidence: {result1.confidence_score:.2%} vs "
                   f"{result2.confidence_score:.2%}). "
                   f"Key difference: {self._find_key_difference(result1, result2)}")
        
        else:
            return (f"{result2.species} is more likely than {result1.species} "
                   f"(confidence: {result2.confidence_score:.2%} vs "
                   f"{result1.confidence_score:.2%}). "
                   f"Key difference: {self._find_key_difference(result2, result1)}")
    
    def _find_key_difference(self,
                            better: SearchExplanation,
                            worse: SearchExplanation) -> str:
        """Find the key factor that differentiates two results"""
        # Find component with biggest difference
        max_diff = 0
        key_factor = ""
        
        for b_comp, w_comp in zip(better.components, worse.components):
            diff = b_comp.contribution - w_comp.contribution
            if diff > max_diff:
                max_diff = diff
                key_factor = b_comp.factor.lower()
        
        return f"better {key_factor}"


# Global explainer instance
_result_explainer = None


def get_result_explainer() -> ResultExplainer:
    """
    Get or create global result explainer
    
    Returns:
        ResultExplainer instance
    """
    global _result_explainer
    
    if _result_explainer is None:
        _result_explainer = ResultExplainer()
    
    return _result_explainer

