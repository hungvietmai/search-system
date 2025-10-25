"""
Script to split the dataset into training, validation, and test sets.
This ensures that the model can learn patterns from one part of the data,
be tuned using another part, and be tested fairly on unseen data.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from app.database import get_db_context
from app.models import LeafImage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset() -> pd.DataFrame:
    """
    Load the complete dataset from the database
    
    Returns:
        DataFrame with dataset information
    """
    logger.info("Loading dataset from database...")
    
    with get_db_context() as db:
        # Query all leaf images
        images = db.query(LeafImage).all()
        
        # Convert to DataFrame
        data = []
        for img in images:
            data.append({
                'file_id': img.file_id,
                'image_path': img.image_path,
                'species': img.species,
                'source': img.source,
                'segmented_path': img.segmented_path
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} images from database")
        logger.info(f"Species count: {df['species'].nunique()}")
        logger.info(f"Lab images: {len(df[df['source'] == 'lab'])}")
        logger.info(f"Field images: {len(df[df['source'] == 'field'])}")
        
        return df


def split_dataset(df: pd.DataFrame, 
                  train_ratio: float = 0.7, 
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets with stratification by species
    
    Args:
        df: DataFrame with dataset information
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Splitting dataset with ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # First, split into train+val and test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_ratio, 
        stratify=df['species'],  # Stratify by species to maintain distribution
        random_state=random_state
    )
    
    # Calculate adjusted validation ratio for the remaining data
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    
    # Split train+val into train and validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_ratio,
        stratify=train_val_df['species'],  # Stratify by species to maintain distribution
        random_state=random_state
    )
    
    logger.info(f"Dataset split:")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"  Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    logger.info(f"  Total: {len(df)}")
    
    # Verify species distribution is maintained
    logger.info(f"Species distribution in train: {train_df['species'].nunique()}")
    logger.info(f"Species distribution in val: {val_df['species'].nunique()}")
    logger.info(f"Species distribution in test: {test_df['species'].nunique()}")
    logger.info(f"Original species distribution: {df['species'].nunique()}")
    
    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, 
                val_df: pd.DataFrame, 
                test_df: pd.DataFrame,
                output_dir: Path):
    """
    Save the dataset splits to CSV files
    
    Args:
        train_df: Training dataset
        val_df: Validation dataset
        test_df: Test dataset
        output_dir: Directory to save the splits
    """
    logger.info(f"Saving dataset splits to {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train_dataset.csv"
    val_path = output_dir / "val_dataset.csv"
    test_path = output_dir / "test_dataset.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved train dataset to {train_path}")
    logger.info(f"Saved validation dataset to {val_path}")
    logger.info(f"Saved test dataset to {test_path}")


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/validation/test sets")
    parser.add_argument("--output-dir", type=str, default="data/splits",
                       help="Directory to save the dataset splits")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Proportion of data for training")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Proportion of data for validation")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Proportion of data for testing")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Load dataset
    df = load_dataset()
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(
        df, 
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state
    )
    
    # Save splits
    output_dir = Path(args.output_dir)
    save_splits(train_df, val_df, test_df, output_dir)
    
    logger.info("Dataset splitting complete!")


if __name__ == "__main__":
    main()