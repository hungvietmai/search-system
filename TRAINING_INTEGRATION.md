# Training System Integration Guide

This guide explains how to use the training system to improve the leaf image search model and integrate the trained model into your search functionality.

## Overview

The training system includes:

- Dataset splitting into train/validation/test sets
- Learning-to-Rank model training with validation monitoring
- Model evaluation on test data
- Integration with the search API

## Step 1: Train the Model

First, split your dataset and train the model:

```bash
# Split the dataset into train/validation/test sets
python scripts/split_dataset.py \
    --output-dir data/splits \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15

# Train the model with validation monitoring
python scripts/train_ltr_model.py \
    --train-data data/splits/train_dataset.csv \
    --val-data data/splits/val_dataset.csv \
    --test-data data/splits/test_dataset.csv \
    --algorithm linear \
    --model-output data/trained_ltr_model.pkl
```

## Step 2: Evaluate the Model

Evaluate the trained model on the test set:

```bash
python scripts/evaluate_model.py \
    --test-data data/splits/test_dataset.csv \
    --model-path data/trained_ltr_model.pkl \
    --output-file evaluation_results.txt
```

## Step 3: Integrate with Search API

The system has been enhanced with a new endpoint that uses the trained model:

### New Search Endpoint

A new endpoint `/search-improved` has been added that:

1. Performs initial similarity search using FAISS
2. Applies the trained Learning-to-Rank model to re-rank results
3. Returns improved search results based on multiple features

Example usage:

```bash
curl -X POST "http://localhost:8000/search-improved?top_k=10" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
 -F "file=@path/to/your/image.jpg"
```

### How It Works

The improved search endpoint works as follows:

1. **Initial Search**: Performs similarity search using FAISS to get top candidates
2. **Feature Generation**: Creates ranking features for each candidate including:
   - Similarity scores (cosine, euclidean distance)
   - Species frequency and popularity
   - Image quality scores
   - Source type (lab vs field)
   - Click-through rates and conversion rates
3. **Re-ranking**: Uses the trained LTR model to re-rank results based on all features
4. **Result**: Returns re-ranked results that should be more relevant

## Step 4: Using the Trained Model in Your Code

If you want to use the trained model programmatically, here's how:

```python
from app.learning_to_rank import get_ltr_engine, LTRAlgorithm
import pickle

# Load the trained model
def load_trained_model(model_path: str = "data/trained_ltr_model.pkl"):
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)

    ltr_engine = get_ltr_engine(algorithm=LTRAlgorithm.LINEAR)

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
        ltr_engine.model.weights = saved_data

    return ltr_engine

# Create ranking features for candidates
from app.learning_to_rank import RankingFeatures
from app.similarity_metrics import MetricCalculator

def create_ranking_features(query_features, candidate_features, candidate_info):
    ranking_features = RankingFeatures(
        vector_similarity=float(MetricCalculator.cosine_similarity(query_features, candidate_features)),
        cosine_similarity=float(MetricCalculator.cosine_similarity(query_features, candidate_features)),
        euclidean_distance=float(MetricCalculator.l2_distance(query_features, candidate_features)),
        species_frequency=candidate_info.get('species_frequency', 0.5),
        species_popularity=candidate_info.get('species_popularity', 0.5),
        image_quality_score=candidate_info.get('image_quality_score', 0.8),
        source_score=1.0 if candidate_info.get('source') == 'lab' else 0.5,
        temporal_score=candidate_info.get('temporal_score', 0.5),
        diversity_score=candidate_info.get('diversity_score', 0.5),
        click_through_rate=candidate_info.get('click_through_rate', 0.1),
        conversion_rate=candidate_info.get('conversion_rate', 0.1)
    )
    return ranking_features

# Re-rank candidates using the trained model
model = load_trained_model()
features_list = [create_ranking_features(query, candidate, info) for candidate, info in zip(candidates, infos)]
ranked_indices = model.model.rank(features_list)
```

## Step 5: Continuous Improvement

To continuously improve the model:

1. **Collect Feedback**: Use the existing feedback endpoints to collect user interaction data
2. **Retrain**: Periodically retrain the model with new data
3. **A/B Testing**: Compare the performance of the improved search vs basic similarity search

## Key Benefits

- **Better Relevance**: The trained model considers multiple factors beyond just visual similarity
- **Adaptability**: The model can learn from user feedback to improve results
- **Performance**: Uses efficient FAISS for initial search, then applies LTR to top candidates
- **Validation**: Includes proper train/validation/test splits to prevent overfitting

## Performance Metrics

The evaluation script provides these metrics:

- **MSE (Mean Squared Error)**: Lower is better
- **MAE (Mean Absolute Error)**: Lower is better
- **NDCG (Normalized Discounted Cumulative Gain)**: Higher is better (0-1 scale)
- **Correlation**: Higher is better (-1 to 1 scale)

## Troubleshooting

If the trained model is not loaded properly:

- Check that `data/trained_ltr_model.pkl` exists
- Verify the model was trained successfully
- Ensure the endpoint falls back gracefully to basic similarity search when the model is unavailable

The system is designed to fall back to basic similarity search if the trained model is not available, ensuring continuous service.
