"""
Relevance Feedback Module
Allows users to provide feedback on search results and improves ranking

Features:
- Explicit feedback (relevant/irrelevant marking)
- Implicit feedback (clicks, dwell time)
- Rocchio algorithm for query expansion
- Online learning integration
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from sqlalchemy.orm import Session

from app.models import UserFeedback

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of user feedback"""
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    CLICKED = "clicked"
    SELECTED = "selected"
    DISMISSED = "dismissed"


@dataclass
class FeedbackEvent:
    """User feedback event"""
    query_image_hash: str
    result_file_id: int
    feedback_type: FeedbackType
    timestamp: datetime
    position: int  # Position in search results
    session_id: Optional[str] = None
    dwell_time: Optional[float] = None  # Time spent viewing (seconds)
    confidence: float = 1.0  # Confidence in feedback (0-1)


class RocchioFeedback:
    """
    Rocchio algorithm for relevance feedback
    Adjusts query vector based on relevant/irrelevant results
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 beta: float = 0.75,
                 gamma: float = 0.15):
        """
        Initialize Rocchio feedback
        
        Args:
            alpha: Weight for original query
            beta: Weight for relevant documents
            gamma: Weight for irrelevant documents
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def adjust_query(self,
                    query_vector: np.ndarray,
                    relevant_vectors: List[np.ndarray],
                    irrelevant_vectors: List[np.ndarray]) -> np.ndarray:
        """
        Adjust query vector using Rocchio algorithm
        
        Formula:
        Q' = α*Q + β*(1/|R|)*Σr - γ*(1/|I|)*Σi
        
        Args:
            query_vector: Original query vector
            relevant_vectors: Vectors of relevant results
            irrelevant_vectors: Vectors of irrelevant results
            
        Returns:
            Adjusted query vector
        """
        adjusted = self.alpha * query_vector
        
        # Add relevant vectors
        if relevant_vectors:
            relevant_centroid = np.mean(relevant_vectors, axis=0)
            adjusted += self.beta * relevant_centroid
        
        # Subtract irrelevant vectors
        if irrelevant_vectors:
            irrelevant_centroid = np.mean(irrelevant_vectors, axis=0)
            adjusted -= self.gamma * irrelevant_centroid
        
        # Normalize
        norm = np.linalg.norm(adjusted)
        if norm > 0:
            adjusted = adjusted / norm
        
        return adjusted


class RelevanceFeedbackEngine:
    """
    Main relevance feedback engine
    Manages feedback collection and query refinement
    """
    
    def __init__(self, db: Session):
        """
        Initialize feedback engine
        
        Args:
            db: Database session
        """
        self.db = db
        self.rocchio = RocchioFeedback()
    
    def record_feedback(self, event: FeedbackEvent) -> int:
        """
        Record user feedback
        
        Args:
            event: Feedback event
            
        Returns:
            Feedback ID
        """
        feedback = UserFeedback(
            query_image_hash=event.query_image_hash,
            query_timestamp=event.timestamp,
            session_id=event.session_id,
            result_file_id=event.result_file_id,
            result_position=event.position,
            feedback_type=event.feedback_type.value,
            confidence=event.confidence,
            dwell_time=event.dwell_time
        )
        
        self.db.add(feedback)
        self.db.commit()
        self.db.refresh(feedback)
        
        logger.info(f"Recorded {event.feedback_type.value} feedback for file {event.result_file_id}")
        
        return int(feedback.id)  # type: ignore
    
    def get_feedback_for_query(self, query_hash: str) -> List[UserFeedback]:
        """
        Get all feedback for a query
        
        Args:
            query_hash: Query image hash
            
        Returns:
            List of feedback records
        """
        feedbacks = self.db.query(UserFeedback).filter(
            UserFeedback.query_image_hash == query_hash
        ).all()
        
        return feedbacks
    
    def get_relevance_judgments(self, 
                                query_hash: str) -> Tuple[List[int], List[int]]:
        """
        Get relevant and irrelevant file IDs for a query
        
        Args:
            query_hash: Query image hash
            
        Returns:
            Tuple of (relevant_file_ids, irrelevant_file_ids)
        """
        feedbacks = self.get_feedback_for_query(query_hash)
        
        relevant_ids = []
        irrelevant_ids = []
        
        for feedback in feedbacks:
            feedback_type = str(feedback.feedback_type)
            if feedback_type == FeedbackType.RELEVANT.value:
                relevant_ids.append(feedback.result_file_id)
            elif feedback_type == FeedbackType.IRRELEVANT.value:
                irrelevant_ids.append(feedback.result_file_id)
            elif feedback_type == FeedbackType.SELECTED.value:
                # Implicit relevance
                relevant_ids.append(feedback.result_file_id)
            elif feedback_type == FeedbackType.DISMISSED.value:
                # Implicit irrelevance
                irrelevant_ids.append(feedback.result_file_id)
        
        return relevant_ids, irrelevant_ids
    
    def refine_query(self,
                    query_vector: np.ndarray,
                    query_hash: str,
                    get_vector_fn: Callable) -> np.ndarray:
        """
        Refine query using Rocchio algorithm
        
        Args:
            query_vector: Original query vector
            query_hash: Query image hash
            get_vector_fn: Function to get feature vector for file_id
            
        Returns:
            Refined query vector
        """
        # Get relevance judgments
        relevant_ids, irrelevant_ids = self.get_relevance_judgments(query_hash)
        
        if not relevant_ids and not irrelevant_ids:
            # No feedback available
            return query_vector
        
        # Get vectors for relevant/irrelevant results
        relevant_vectors = []
        for file_id in relevant_ids:
            vector = get_vector_fn(file_id)
            if vector is not None:
                relevant_vectors.append(vector)
        
        irrelevant_vectors = []
        for file_id in irrelevant_ids:
            vector = get_vector_fn(file_id)
            if vector is not None:
                irrelevant_vectors.append(vector)
        
        # Apply Rocchio
        refined_vector = self.rocchio.adjust_query(
            query_vector,
            relevant_vectors,
            irrelevant_vectors
        )
        
        logger.info(f"Refined query using {len(relevant_vectors)} relevant and "
                   f"{len(irrelevant_vectors)} irrelevant results")
        
        return refined_vector
    
    def get_click_through_rate(self, file_id: int) -> float:
        """
        Get click-through rate for a file
        
        Args:
            file_id: File ID
            
        Returns:
            CTR (0-1)
        """
        # Count impressions (appeared in results)
        total = self.db.query(UserFeedback).filter(
            UserFeedback.result_file_id == file_id
        ).count()
        
        if total == 0:
            return 0.0
        
        # Count clicks
        clicks = self.db.query(UserFeedback).filter(
            UserFeedback.result_file_id == file_id,
            UserFeedback.feedback_type.in_([
                FeedbackType.CLICKED.value,
                FeedbackType.SELECTED.value
            ])
        ).count()
        
        return clicks / total
    
    def get_conversion_rate(self, file_id: int) -> float:
        """
        Get conversion rate (selections) for a file
        
        Args:
            file_id: File ID
            
        Returns:
            Conversion rate (0-1)
        """
        # Count impressions
        total = self.db.query(UserFeedback).filter(
            UserFeedback.result_file_id == file_id
        ).count()
        
        if total == 0:
            return 0.0
        
        # Count selections
        selections = self.db.query(UserFeedback).filter(
            UserFeedback.result_file_id == file_id,
            UserFeedback.feedback_type == FeedbackType.SELECTED.value
        ).count()
        
        return selections / total
    
    def get_average_position(self, file_id: int) -> float:
        """
        Get average position in search results
        
        Args:
            file_id: File ID
            
        Returns:
            Average position (1-based)
        """
        from sqlalchemy import func
        
        result = self.db.query(
            func.avg(UserFeedback.result_position)
        ).filter(
            UserFeedback.result_file_id == file_id
        ).scalar()
        
        return result or 0.0
    
    def get_training_data(self, 
                         limit: int = 1000) -> Tuple[List[str], List[List[int]], List[List[float]]]:
        """
        Get training data for learning-to-rank
        
        Args:
            limit: Maximum number of queries
            
        Returns:
            Tuple of (query_hashes, file_ids_per_query, relevance_scores_per_query)
        """
        from sqlalchemy import func, distinct
        
        # Get unique queries with feedback
        queries = self.db.query(
            distinct(UserFeedback.query_image_hash)
        ).limit(limit).all()
        
        query_hashes = [q[0] for q in queries]
        file_ids_per_query = []
        relevance_scores_per_query = []
        
        for query_hash in query_hashes:
            # Get feedback for this query
            feedbacks = self.db.query(UserFeedback).filter(
                UserFeedback.query_image_hash == query_hash
            ).order_by(UserFeedback.result_position).all()
            
            # Aggregate by file_id
            file_scores = {}
            for feedback in feedbacks:
                file_id = feedback.result_file_id
                if file_id not in file_scores:
                    file_scores[file_id] = 0.0
                
                # Compute relevance score from feedback
                feedback_type = str(feedback.feedback_type)
                if feedback_type == FeedbackType.RELEVANT.value:
                    file_scores[file_id] += 2.0
                elif feedback_type == FeedbackType.SELECTED.value:
                    file_scores[file_id] += 1.5
                elif feedback_type == FeedbackType.CLICKED.value:
                    file_scores[file_id] += 0.5
                elif feedback_type == FeedbackType.IRRELEVANT.value:
                    file_scores[file_id] -= 1.0
                elif feedback_type == FeedbackType.DISMISSED.value:
                    file_scores[file_id] -= 0.5
            
            if file_scores:
                file_ids = list(file_scores.keys())
                scores = [file_scores[fid] for fid in file_ids]
                
                file_ids_per_query.append(file_ids)
                relevance_scores_per_query.append(scores)
        
        logger.info(f"Retrieved training data for {len(query_hashes)} queries")
        
        return query_hashes, file_ids_per_query, relevance_scores_per_query


def get_relevance_feedback_engine(db: Session) -> RelevanceFeedbackEngine:
    """
    Get relevance feedback engine
    
    Args:
        db: Database session
        
    Returns:
        RelevanceFeedbackEngine instance
    """
    return RelevanceFeedbackEngine(db)

