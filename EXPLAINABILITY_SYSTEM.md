# Search Result Explainability System

This document explains the components, calculations, and formulas used in the search result explainability system for the leaf image identification application.

## Overview

The explainability system provides transparency for search results by breaking down the confidence score into multiple contributing factors. Each search result includes an explanation of why it was matched, how confident the system is, and what factors contributed to that confidence.

## Explanation Components

Each search result explanation contains several key components:

### 1. Visual Similarity

- **Factor**: Visual Similarity
- **Score**: Based on L2 distance between feature vectors
- **Weight**: 50% (0.5)
- **Description**: Quality of visual match based on distance
- **Contribution**: Score × Weight

**Calculation**:

- Converts L2 distance to similarity using exponential decay: `similarity = exp(-k * distance)` where k=1.0
- Distance thresholds:
  - Excellent: < 0.4 (similarity ~0.67)
  - Good: < 0.7 (similarity ~0.49)
  - Fair: < 1.1 (similarity ~0.33)
  - Poor: < 1.6 (similarity ~0.20)

### 2. Search Ranking

- **Factor**: Search Ranking
- **Score**: Based on position in results
- **Weight**: 20% (0.2)
- **Contribution**: Score × Weight

**Calculation**:

- `rank_score = 1.0 - (position / max(total_results, 10))`
- Higher positions (closer to 0) get higher scores
- Position 0: "Top match among all results"
- Positions 1-2: "Ranked #X, high confidence"
- Positions 3-4: "Ranked #X, good match"
- Positions 5+: "Ranked #X"

### 3. Species Data Quality

- **Factor**: Species Data Quality
- **Score**: Based on reference images count
- **Weight**: 20% (0.2)
- **Contribution**: Score × Weight

**Calculation**:

- Count of images for the species in the database
- > 100 images: Score 1.0 ("Well-represented species (X+ reference images)")
- > 50 images: Score 0.8 ("Common species (X reference images)")
- > 20 images: Score 0.6 ("Moderately represented (X reference images)")
- > 10 images: Score 0.4 ("Limited reference data (X images)")
- ≤10 images: Score 0.2 ("Rare species with limited data (X images)")

### 4. Image Quality

- **Factor**: Image Quality
- **Score**: Based on image source
- **Weight**: 10% (0.1)
- **Contribution**: Score × Weight

**Calculation**:

- Lab photographs: Score 1.0 ("High-quality lab photograph (controlled conditions)")
- Field photographs: Score 0.7 ("Field photograph (natural conditions)")

## Formula for Overall Confidence Score

The overall confidence score is calculated as a weighted sum of all components:

```
confidence_score = Σ(component.contribution for component in components)
confidence_score = Σ(component.score * component.weight for component in components)
```

## Confidence Level Categories

Based on the overall confidence score, results are categorized as:

- Very High: 90-100% (≥0.90)
- High: 75-90% (≥0.75)
- Medium: 50-75% (≥0.50)
- Low: 25-50% (≥0.25)
- Very Low: 0-25% (<0.25)

## Visual Feature Analysis

The system analyzes visual features by:

1. Dividing feature vectors into 4 regions:
   - Overall shape (0 to 1/4 of features)
   - Texture patterns (1/4 to 1/2 of features)
   - Edge characteristics (1/2 to 3/4 of features)
   - Color distribution (3/4 to end of features)
2. Computing cosine similarity for each region
3. Identifying regions with high similarity (>0.85 = "Very similar", >0.70 = "Similar")

## Potential Concerns

The system identifies potential concerns when:

- Confidence < 0.50: "Low overall confidence"
- Distance > 1.0: "Substantial visual differences"
- Position > 5: "Not among top matches"
- Species count < 20: "Limited reference data"

## Example Breakdown

From the example provided:

- Visual Similarity: Score 0.5897, Weight 0.5, Contribution 0.2948
- Search Ranking: Score 0.6, Weight 0.2, Contribution 0.12
- Species Data Quality: Score 1.0, Weight 0.2, Contribution 0.2
- Image Quality: Score 0.7, Weight 0.1, Contribution 0.07
- Overall Confidence: 0.6848 (Medium confidence level)
