"""
Utility functions for the Fingerprint Recognition System
Includes helper functions for file operations and data handling
"""

import os
import cv2
import numpy as np
from typing import List, Tuple


def load_image(image_path: str, target_size: Tuple[int, int] = (96, 96)) -> np.ndarray:
    """
    Load and preprocess a fingerprint image
    
    LINEAR ALGEBRA CONCEPT: Images are represented as 2D matrices
    - Each pixel value is an element in the matrix
    - Grayscale conversion creates a single-channel matrix
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)
    
    Returns:
        Preprocessed image as numpy array (matrix)
    """
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale - reduces 3D tensor (RGB) to 2D matrix
    # This is a linear transformation: gray = 0.299*R + 0.587*G + 0.114*B
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to standard dimensions
    # This operation maintains the matrix structure but changes dimensions
    img_resized = cv2.resize(img_gray, target_size)
    
    # Normalize pixel values to [0, 1] range
    # Matrix scalar division: each element divided by 255
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return img_normalized


def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image files from a directory
    
    Args:
        directory: Directory path to search
        extensions: List of valid extensions (default: common image formats)
    
    Returns:
        List of full paths to image files
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    
    if not os.path.exists(directory):
        return []
    
    image_files = []
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in extensions):
            image_files.append(os.path.join(directory, filename))
    
    return sorted(image_files)


def flatten_image(image: np.ndarray) -> np.ndarray:
    """
    Flatten a 2D image matrix into a 1D feature vector
    
    LINEAR ALGEBRA CONCEPT: Matrix vectorization
    - Converts 2D matrix (h × w) into 1D vector (h*w × 1)
    - This is necessary for many ML algorithms that work with vectors
    - Example: 96×96 image becomes 9,216-dimensional vector
    
    Args:
        image: 2D image matrix
    
    Returns:
        1D feature vector
    """
    # Reshape from (height, width) to (height * width,)
    # This is a linear transformation preserving all information
    return image.flatten()


def euclidean_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors
    
    LINEAR ALGEBRA CONCEPT: L2 Norm (Euclidean distance)
    - Distance = ||v1 - v2|| = sqrt(sum((v1_i - v2_i)^2))
    - This is the geometric distance in n-dimensional space
    - Measures dissimilarity: smaller distance = more similar
    
    Mathematical formula:
    d(x, y) = √(Σ(x_i - y_i)²)
    
    Args:
        vector1: First feature vector
        vector2: Second feature vector
    
    Returns:
        Euclidean distance (scalar value)
    """
    # Calculate difference vector: v1 - v2
    diff = vector1 - vector2
    
    # Calculate squared Euclidean distance: ||diff||²
    # This is the dot product of diff with itself: diff · diff
    squared_distance = np.dot(diff, diff)
    
    # Return the Euclidean distance: sqrt(||diff||²)
    return np.sqrt(squared_distance)


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors
    
    LINEAR ALGEBRA CONCEPT: Cosine of angle between vectors
    - Similarity = (v1 · v2) / (||v1|| × ||v2||)
    - Measures angle between vectors, not magnitude
    - Range: [-1, 1], where 1 = identical direction
    
    Mathematical formula:
    cos(θ) = (x · y) / (||x|| × ||y||)
    
    Args:
        vector1: First feature vector
        vector2: Second feature vector
    
    Returns:
        Cosine similarity score
    """
    # Calculate dot product: v1 · v2
    dot_product = np.dot(vector1, vector2)
    
    # Calculate norms: ||v1|| and ||v2||
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Return cosine similarity
    return dot_product / (norm1 * norm2)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length
    
    LINEAR ALGEBRA CONCEPT: Vector normalization
    - Normalized vector = v / ||v||
    - Results in a unit vector (length = 1) pointing in same direction
    - Useful for comparing directions regardless of magnitude
    
    Args:
        vector: Input vector
    
    Returns:
        Normalized unit vector
    """
    norm = np.linalg.norm(vector)
    
    if norm == 0:
        return vector
    
    # Scalar division: each element divided by the norm
    return vector / norm


def create_sample_fingerprint(output_path: str, size: Tuple[int, int] = (96, 96)):
    """
    Create a synthetic fingerprint-like pattern for testing
    
    Args:
        output_path: Path to save the generated image
        size: Size of the image (width, height)
    """
    # Create a blank image
    img = np.zeros(size, dtype=np.uint8)
    
    # Add some ridge-like patterns
    for i in range(0, size[0], 3):
        cv2.line(img, (i, 0), (i, size[1]), 255, 1)
    
    # Add some noise for variation
    noise = np.random.randint(0, 50, size, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # Save the image
    cv2.imwrite(output_path, img)


def print_matrix_info(matrix: np.ndarray, name: str = "Matrix"):
    """
    Print information about a matrix for debugging
    
    Args:
        matrix: The matrix to analyze
        name: Name to display
    """
    print(f"\n{name} Information:")
    print(f"  Shape: {matrix.shape}")
    print(f"  Data type: {matrix.dtype}")
    print(f"  Min value: {np.min(matrix):.4f}")
    print(f"  Max value: {np.max(matrix):.4f}")
    print(f"  Mean value: {np.mean(matrix):.4f}")
    print(f"  Standard deviation: {np.std(matrix):.4f}")
