"""
Complete training pipeline script
This script runs the entire training workflow: dataset splitting, model training, and evaluation
"""
import sys
from pathlib import Path
import subprocess
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: str) -> bool:
    """
    Run a command and return success status
    
    Args:
        cmd: Command to run
        
    Returns:
        True if command succeeded, False otherwise
    """
    logger.info(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info("Command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False


def main():
    logger.info("Starting complete training pipeline...")
    
    # Step 1: Split the dataset
    logger.info("Step 1: Splitting dataset into train/validation/test sets")
    split_cmd = "python scripts/split_dataset.py --output-dir data/splits --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15"
    if not run_command(split_cmd):
        logger.error("Dataset splitting failed!")
        return False
    
    # Step 2: Train the model
    logger.info("Step 2: Training the Learning-to-Rank model")
    train_cmd = "python scripts/train_ltr_model.py --train-data data/splits/train_dataset.csv --val-data data/splits/val_dataset.csv --test-data data/splits/test_dataset.csv --algorithm linear --model-output data/trained_ltr_model.pkl"
    if not run_command(train_cmd):
        logger.error("Model training failed!")
        return False
    
    # Step 3: Evaluate the model
    logger.info("Step 3: Evaluating the trained model")
    eval_cmd = "python scripts/evaluate_model.py --test-data data/splits/test_dataset.csv --model-path data/trained_ltr_model.pkl --output-file evaluation_results.txt"
    if not run_command(eval_cmd):
        logger.error("Model evaluation failed!")
        return False
    
    logger.info("Training pipeline completed successfully!")
    logger.info("Results:")
    logger.info("- Dataset split into train/validation/test sets")
    logger.info("- Model trained with validation monitoring")
    logger.info("- Final model evaluated on test set")
    logger.info("- Results saved to evaluation_results.txt")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Pipeline completed successfully!")
    else:
        logger.error("Pipeline failed!")
        sys.exit(1)