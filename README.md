# Fingerprint Recognition System

A Python-based fingerprint recognition system demonstrating linear algebra concepts in biometric authentication.

## Overview

This project implements a fingerprint matching system using:

- **Principal Component Analysis (PCA)** for dimensionality reduction
- **Euclidean Distance** for similarity measurement
- **Vector operations** for feature extraction
- Real fingerprint dataset (SOCOFing - Sokoto Coventry Fingerprint Dataset subset)

## Linear Algebra Concepts Applied

1. **Matrix Operations**: Fingerprint images are represented as matrices
2. **Eigenvectors & Eigenvalues**: PCA uses these for dimensionality reduction
3. **Vector Norms**: Euclidean distance for comparing fingerprints
4. **Matrix Decomposition**: Covariance matrix analysis
5. **Dot Products**: For projection onto principal components

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone or navigate to the project directory**:

   ```bash
   cd fingerprint-recognition
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The project includes realistic fingerprint samples with proper labeling:

### Dataset Structure:

- **Database** (`dataset/db/`): 10 fingerprint images from 5 subjects
  - Naming: `SubjectID_FingerType_SampleNumber.png`
  - Example: `Subject001_RightThumb_01.png`
- **Query** (`dataset/query/`): 3 test fingerprints
  - Naming: `Query_SubjectID_FingerType.png`
  - Example: `Query_Subject001_RightThumb.png`

### Subjects:

- **Subject001**: Right Thumb, Right Index
- **Subject002**: Right Thumb, Left Index
- **Subject003**: Right Thumb, Right Middle
- **Subject004**: Left Thumb, Left Index
- **Subject005**: Right Thumb, Right Ring

### Regenerate Dataset:

To create fresh realistic fingerprint samples:

```bash
python setup_real_dataset.py
```

### Use Your Own Fingerprints:

To use real fingerprint scans or public datasets:

1. Download fingerprint images (e.g., from SOCOFing, FVC2000, or NIST databases)
2. Place images in `dataset/db/` and `dataset/query/`
3. Use consistent naming for tracking matches

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

1. **Load Dataset**: Fingerprint images are loaded from the dataset folder
2. **Preprocessing**: Images are converted to grayscale and resized
3. **Feature Extraction**:
   - Images are flattened into vectors
   - PCA reduces dimensionality (Linear Algebra!)
4. **Matching**:
   - Select a query fingerprint
   - System calculates Euclidean distance to all database fingerprints
   - Returns closest matches
5. **Visualization**: Simple Tkinter GUI displays results

## Project Structure

```
fingerprint-recognition/
├── src/
│   ├── main.py                    # Main application entry point
│   ├── fingerprint_processor.py   # Core processing with linear algebra
│   ├── gui.py                     # Simple Tkinter frontend
│   └── utils.py                   # Utility functions
├── dataset/
│   ├── db/                        # Database fingerprints
│   └── query/                     # Query fingerprints for testing
├── requirements.txt               # Python dependencies
├── run.ps1                        # PowerShell run script
├── run.bat                        # Batch run script
└── README.md                      # This file
```

## Usage Guide

1. **Launch the application** using one of the methods above
2. **Select a query fingerprint** from the dropdown menu
3. **Click "Match Fingerprint"** to find similar prints
4. **View results**: Top 3 matches with similarity scores
5. **Experiment**: Try different query images to see matching results

## Linear Algebra Annotations

The code is heavily annotated, especially in `fingerprint_processor.py`, where linear algebra concepts are applied:

- **Matrix representation** of images
- **Covariance matrix** computation
- **Eigendecomposition** for PCA
- **Vector projections** onto principal components
- **Norm calculations** for distance metrics

## Technical Details

- **Image Size**: 96x96 pixels (converted to grayscale)
- **Feature Vector**: Original 9,216 dimensions
- **PCA Components**: Reduced to 50 dimensions (configurable)
- **Distance Metric**: Euclidean distance (L2 norm)

## Limitations

- This is an educational project demonstrating concepts
- Real-world fingerprint systems use more sophisticated algorithms (minutiae extraction, ridge patterns)
- Small dataset for demonstration purposes
- No authentication or security features

## Future Enhancements

- Add minutiae-based matching
- Implement more distance metrics (cosine similarity, Manhattan distance)
- Expand dataset
- Add fingerprint quality assessment
- Real-time camera capture

## References

- SOCOFing Dataset: Sokoto Coventry Fingerprint Dataset
- Principal Component Analysis (PCA)
- Eigenface technique adapted for fingerprints

## License

Educational use only. Dataset images are from public domain sources.

## Troubleshooting

**Issue**: Module not found errors

- **Solution**: Run `pip install -r requirements.txt`

**Issue**: No fingerprints found

- **Solution**: Ensure dataset folder contains images

**Issue**: GUI doesn't appear

- **Solution**: Check Tkinter is installed (usually comes with Python)

## Contact

For questions or issues, please refer to the code annotations or create an issue in the repository.
