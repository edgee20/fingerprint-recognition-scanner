# Fingerprint Recognition System

A Python-based fingerprint recognition system demonstrating linear algebra concepts in biometric authentication.

## Overview

This project implements a fingerprint matching system using:

- **Principal Component Analysis (PCA)** for dimensionality reduction
- **Multi-scale feature extraction** (global, gradient, and local features)
- **Multiple distance metrics** (L2, L1, Cosine similarity, Correlation)
- **Gradient-based features** using Sobel operators
- Real fingerprint dataset (FVC2002 Database subset)

## Linear Algebra Concepts Applied

1. **Matrix Operations**: Fingerprint images represented as matrices
2. **Eigenvectors & Eigenvalues**: PCA for dimensionality reduction
3. **Vector Norms**: L1 and L2 distance metrics
4. **Matrix Decomposition**: Covariance matrix eigendecomposition
5. **Dot Products**: Projection onto principal components
6. **Convolution**: Sobel operators for gradient extraction
7. **Multiple Subspaces**: Separate PCA models for different feature types

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone or navigate to the project directory**:

   ```bash
   cd fingerprint-recognition-scanner
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

This project uses real fingerprint images from the FVC2002 (Fingerprint Verification Competition) database.

### Dataset Structure:

- **Database** (`dataset/db/`): 32 fingerprint images from 4 subjects
  - Naming: `PersonID_ScanNumber.tif`
  - Example: `101_1.tif`, `101_2.tif`, etc.
  - 8 scans per person (different impressions of the same finger)
- **Query** (`dataset/query/`): Test fingerprints for matching
  - Contains sample scans from the database subjects

### Subjects in Database:

- Person 101: 8 scans
- Person 102: 8 scans
- Person 103: 8 scans
- Person 104: 8 scans

The system correctly matches all 8 scans from the same person when given a query fingerprint.

## Running the Application

### Method 1: Using Python directly

```bash
python src/main.py
```

### Method 2: Using PowerShell script (Windows)

```powershell
.\run.ps1
```

### Method 3: Using batch file (Windows)

```cmd
run.bat
```

## How It Works

1. **Load Dataset**: Fingerprint images loaded from dataset folder
2. **Preprocessing**:
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Bilateral filtering for edge preservation
3. **Multi-Scale Feature Extraction**:
   - Global features: Flattened image matrix
   - Gradient features: Sobel operators for edge detection
   - Local features: Histogram of oriented gradients in cells
4. **PCA Training**:
   - Three separate PCA models trained on different feature types
   - Reduces 65,536 + 65,536 + 1,800 dimensions to 32 + 30 + 32 = 94 dimensions
5. **Matching**:
   - Combines 4 similarity metrics (L2, L1, Cosine, Correlation)
   - Returns all scans from the same person, sorted by similarity
6. **Visualization**: Tkinter GUI with scrollable results

## Project Structure

```
fingerprint-recognition-scanner/
├── src/
│   ├── main.py                  # Application entry point
│   ├── enhanced_processor.py    # Multi-scale PCA with gradient features
│   ├── gui.py                   # Tkinter interface
│   └── utils.py                 # Helper functions
├── dataset/
│   ├── db/                      # 32 fingerprint images (4 subjects, 8 scans each)
│   └── query/                   # Query fingerprints for testing
├── requirements.txt             # Dependencies
├── run.ps1                      # PowerShell launcher
├── run.bat                      # Batch launcher
└── README.md                    # Documentation
```

## Usage Guide

1. Launch the application using one of the methods above
2. Select a query fingerprint from the dropdown menu
3. Click "Match Fingerprint" to find similar prints
4. View results: All matching scans from the same person with similarity scores
5. Try different query images to see how the system identifies all impressions from the same finger

## Linear Algebra Implementation

The code includes detailed annotations in `enhanced_processor.py` explaining:

- Matrix representation of fingerprint images
- Covariance matrix computation
- Eigendecomposition for multiple PCA models
- Gradient vectors using Sobel operators (convolution matrices)
- Vector projections onto principal component subspaces
- Multiple norm calculations (L1, L2, dot products)

## Technical Details

- **Image Size**: 256x256 pixels (grayscale)
- **Feature Dimensions**:
  - Global: 65,536 → 32 via PCA
  - Gradient: 65,536 → 30 via PCA
  - Local: 1,800 → 32 via PCA
  - Combined: 94 dimensions
- **Distance Metrics**: Weighted combination of L2, L1, Cosine similarity, and Correlation
- **Dataset**: FVC2002 real fingerprint images

## Limitations

- Educational project demonstrating linear algebra concepts
- Real-world systems may use minutiae extraction and ridge pattern analysis
- Limited dataset (32 images from 4 subjects)
- No security or authentication features included

## Future Enhancements

- Add minutiae detection algorithms
- Expand dataset with more subjects
- Implement fingerprint quality assessment
- Add real-time capture capability

## References

- FVC2002: Fingerprint Verification Competition 2002 Database
- Principal Component Analysis (PCA)
- Sobel operators for edge detection
- Multi-scale feature extraction techniques

## License

Educational use only. FVC2002 dataset used for academic purposes.

## Troubleshooting

**Module not found errors**

- Run `pip install -r requirements.txt`

**No fingerprints found**

- Ensure dataset/db/ and dataset/query/ folders contain .tif images

**GUI doesn't appear**

- Tkinter is usually included with Python, reinstall Python if needed

