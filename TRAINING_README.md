# Training System for Leaf Image Search

This project implements a machine learning training system for improving the leaf image search model. The system follows the standard machine learning workflow of training on one dataset, validating on another, and testing on unseen data.

## Overview

The training system includes:

- Dataset splitting into train/validation/test sets
- Learning-to-Rank model training with validation monitoring
- Model evaluation on test data
- Performance metrics tracking

## Dataset Splitting

The system splits the dataset into three parts:

1. **Training Data (70%)** - Used to train the model and learn patterns
2. **Validation Data (15%)** - Used to tune hyperparameters and prevent overfitting
3. **Test Data (15%)** - Used to evaluate final model performance on unseen data

The splitting is done with stratification by species to maintain the same distribution across all splits.

## Training Process

The training process follows these steps:

1. **Data Preparation**: Load and split the dataset
2. **Feature Generation**: Create ranking features for each image
3. **Model Training**: Train the Learning-to-Rank model using training data
4. **Validation**: Monitor performance on validation data to prevent overfitting
5. **Evaluation**: Test the final model on unseen test data

## Available Algorithms

The system supports multiple Learning-to-Rank algorithms:

- **Linear**: Weighted linear combination of features
- **RankNet**: Pairwise ranking approach
- **LambdaMART**: Gradient boosted trees (coming soon)
- **Listwise**: Listwise ranking (coming soon)

## Usage

### 1. Split the Dataset

```bash
python scripts/split_dataset.py --output-dir data/splits --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --random-state 42
```

### 2. Train the Model

```bash
python scripts/train_ltr_model.py --train-data data/splits/train_dataset.csv --val-data data/splits/val_dataset.csv --test-data data/splits/test_dataset.csv --algorithm linear --model-output data/trained_ltr_model.pkl
```

### 3. Evaluate the Model

```bash
python scripts/evaluate_model.py --test-data data/splits/test_dataset.csv  --model-path data/trained_ltr_model.pkl --output-file evaluation_results.txt
```

## Key Features

### Train-Validation-Test Split

- Ensures model learns patterns from training data
- Allows tuning with validation data
- Provides fair evaluation on unseen test data

### Validation Monitoring

- Prevents overfitting by monitoring validation loss
- Saves the best model based on validation performance
- Provides training and validation loss curves

### Comprehensive Evaluation

- Multiple metrics: MSE, MAE, NDCG, Correlation
- Detailed performance analysis
- Results saved to file for review

## File Structure

```
data/
├── splits/
│   ├── train_dataset.csv
│   ├── val_dataset.csv
│   └── test_dataset.csv
└── trained_ltr_model.pkl

scripts/
├── split_dataset.py          # Split dataset into train/val/test
├── train_ltr_model.py        # Train the LTR model
└── evaluate_model.py         # Evaluate the trained model

app/
└── learning_to_rank.py       # Core LTR implementation with validation support
```

## Metrics Explained

- **MSE (Mean Squared Error)**: Lower is better; measures average squared difference between predicted and actual relevance
- **MAE (Mean Absolute Error)**: Lower is better; measures average absolute difference
- **NDCG (Normalized Discounted Cumulative Gain)**: Higher is better; measures ranking quality
- **Correlation**: Higher is better; measures rank correlation between predictions and actual relevance

## Best Practices

1. **Always use the train-validation-test split** to properly evaluate model performance
2. **Monitor validation metrics** during training to prevent overfitting
3. **Evaluate on unseen test data** for final performance assessment
4. **Compare different algorithms** to find the best performing one for your data
5. **Use stratified splitting** to maintain species distribution across splits

## Extending the System

The system is designed to be extensible:

- Add new ranking features to `RankingFeatures` class
- Implement new LTR algorithms in `learning_to_rank.py`
- Add new evaluation metrics to the evaluation script
- Modify feature generation in the training/evaluation scripts
