# Advanced Image Preprocessing System

This document explains the preprocessing components, techniques, and methods used in the leaf image identification application to prepare images for feature extraction and search.

## Overview

The preprocessing system provides sophisticated image enhancement and normalization to improve the quality of leaf images before feature extraction. It includes multiple techniques for background removal, rotation correction, edge enhancement, and adaptive processing based on image characteristics.

## Preprocessing Components

### 1. Background Removal Methods

The system offers multiple background removal techniques:

#### A. GrabCut Algorithm

- **Method**: `BackgroundRemovalMethod.GRABCUT`
- **Approach**: Interactive segmentation using color and texture models
- **Process**:

1. Uses edge detection to find contours and determine leaf boundaries
2. Creates a bounding rectangle around the main object
3. Builds foreground and background models
4. Iteratively refines the segmentation (10-15 iterations)
5. Applies morphological operations to clean up the mask

- **Fallback**: Uses Otsu thresholding if GrabCut fails or image is too uniform

#### B. Otsu's Thresholding

- **Method**: `BackgroundRemovalMethod.OTSU`
- **Approach**: Automatic threshold selection based on image histogram
- **Process**:
  1. Converts image to grayscale

2. Applies Gaussian blur to reduce noise
3. Calculates optimal threshold using Otsu's method
4. Creates binary mask
5. Applies morphological operations for cleanup

#### C. K-Means Clustering

- **Method**: `BackgroundRemovalMethod.K_MEANS`
- **Approach**: Color-based clustering to separate foreground and background
- **Process**:

1. Reshapes image pixels into feature vectors
2. Performs K-means clustering (k=3 by default)
3. Identifies largest cluster as background
4. Creates mask excluding background cluster

#### D. Adaptive Method

- **Method**: `BackgroundRemovalMethod.ADAPTIVE`
- **Approach**: Tries multiple methods and selects the best one
- **Process**:
  1. Attempts GrabCut and K-means methods

2. Evaluates foreground ratio for each method
3. Selects method with foreground ratio closest to 0.3-0.7 (reasonable leaf size)

### 2. Rotation Detection and Correction

#### Multi-Point Rotation Detection

- **Approach**: Uses multiple feature detection methods for robust angle estimation
- **Methods**:
  1. **Corner-based**: Detects Harris corners and computes principal axis using PCA
  2. **Contour-based**: Fits ellipse to leaf contour and uses ellipse orientation
  3. **Bounding box**: Uses oriented bounding rectangle angle
- **Process**:

1. Applies weighted voting from all methods
2. Computes weighted average of detected angles
3. Rotates image to normalize orientation

#### Rotation Correction

- **Process**: Uses OpenCV's `warpAffine` with rotation matrix
- **Background**: Uses white background to match dataset format

### 3. Enhancement Techniques

#### Edge Enhancement

- **Method**: Unsharp masking using PIL
- **Parameters**: Radius=2, Percent=150, Threshold=3
- **Purpose**: Enhances leaf edges for better feature extraction

#### CLAHE (Contrast Limited Adaptive Histogram Equalization)

- **Method**: Applied to L channel in LAB color space
- **Parameters**: Clip limit=2.0, Tile grid size=(16, 16)
- **Purpose**: Improves contrast while avoiding over-amplification of noise

#### Color Balancing

- **Method**: Gray World assumption
- **Process**:

1. Calculates average for each RGB channel
2. Computes global average
3. Balances each channel to global average

- **Purpose**: Normalizes lighting conditions

#### Denoising

- **Method**: Non-local means denoising
- **Parameters**: h=10, templateWindowSize=7, searchWindowSize=21
- **Purpose**: Preserves edges while removing noise

### 4. Quality Assessment

The system evaluates image quality using multiple metrics:

#### Blur Detection

- **Method**: Laplacian variance
- **Threshold**: <100 indicates blurry image
- **Purpose**: Determines if denoising is needed

#### Brightness Analysis

- **Method**: Mean pixel value in grayscale
- **Ranges**:
  - <80: Dark image
  - > 200: Overexposed image

#### Contrast Measurement

- **Method**: Standard deviation of grayscale image
- **Threshold**: <30 indicates low contrast

#### Sharpness Evaluation

- **Method**: Edge density (Canny edge detection)
- **Calculation**: Ratio of edge pixels to total pixels

### 5. Adaptive Preprocessing

Based on quality assessment, the system applies different enhancements:

- **Blurry images** (<150 blur score): Apply denoising
- **Poor lighting** (dark or overexposed): Apply color balancing
- **Low contrast** (<30): Apply CLAHE enhancement

### 6. Profile-Based Processing

The system offers different processing profiles:

#### Lab Profile

- **Characteristics**: High-quality lab images with clean backgrounds
- **Processing**:
  - Minimal background removal (often skipped)
  - No rotation correction
  - Edge enhancement enabled
  - Adaptive preprocessing enabled

#### Field Profile

- **Characteristics**: Field images with complex backgrounds
- **Processing**:
  - Aggressive background removal (adaptive method)
  - Rotation correction enabled
  - Edge enhancement enabled
  - Adaptive preprocessing enabled

#### Auto Profile

- **Process**: Automatically detects profile based on:
  - Edge density (field images have more edges)
  - Color variance (lab images have more uniform background)
  - If edge density < 0.15 and color std < 50 → Lab profile
  - Otherwise → Field profile

### 7. Advanced Preprocessing Pipeline

The system includes a sophisticated pipeline with:

#### Deep Background Removal

- **Approach**: Multi-stage segmentation inspired by deep learning
- **Stages**:
  1. Coarse segmentation using color, texture, and edge cues
  2. Refinement using GrabCut with probability map

3. Post-processing with morphological operations
4. Largest connected component selection

#### Leaf Characteristic Detection

- **Detected features**:
- Leaf type (simple, compound, lobed, serrated, smooth)
- Aspect ratio
- Complexity (edge complexity metric)
- Symmetry (bilateral symmetry score)
- Color profile (dominant color)
- Texture score

#### Adaptive Parameters

- **Based on leaf type**:
  - Lobed: More edge enhancement, less smoothing
  - Serrated: Moderate edge enhancement
  - Others: Standard enhancement
- **Based on texture**: Adjusts denoising strength
- **Based on aspect ratio**: Adjusts rotation sensitivity

## Preprocessing Workflow

The complete preprocessing workflow follows these steps:

1. **Profile Detection**: Auto-detect if needed
2. **Quality Assessment**: Evaluate image quality metrics
3. **Adaptive Preprocessing**: Apply quality-based enhancements
4. **Background Removal**: Remove background based on profile
5. **Rotation Correction**: Detect and correct orientation
6. **Edge Enhancement**: Enhance leaf edges
7. **Final Enhancement**: Apply CLAHE
8. **Resizing**: Resize to target size (224x224) using LANCZOS resampling

## Performance Improvements

The advanced preprocessing system is expected to provide:

- Background removal: +30-40% accuracy on field images
- Rotation correction: +15-20% accuracy
- Leaf-aware processing: +10-15% accuracy
- Adaptive parameters: Progressive improvement over time
