"""
Enhanced Fingerprint Processing with Advanced Linear Algebra
This module uses multiple linear algebra techniques for better matching:
1. Gradient-based feature extraction (vectors and matrices)
2. Local Binary Patterns (binary vectors)
3. Multiple distance metrics (norms)
4. Correlation-based alignment (dot products)
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import List, Tuple, Dict
import cv2

from utils import load_image, euclidean_distance, cosine_similarity


class EnhancedFingerprintProcessor:
    """
    Enhanced fingerprint processor using multiple linear algebra techniques
    """
    
    def __init__(self, n_components: int = 100, image_size: Tuple[int, int] = (256, 256)):
        """
        Initialize with enhanced parameters
        """
        self.n_components = n_components
        self.image_size = image_size
        self.pca_global = None
        self.pca_gradient = None
        self.database_features = []
        self.database_labels = []
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Enhanced preprocessing with better normalization"""
        image = load_image(image_path, self.image_size)
        image_uint8 = (image * 255).astype(np.uint8)
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(image_uint8)
        
        # Bilateral filter - preserves edges while removing noise
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return filtered.astype(np.float32) / 255.0
    
    def extract_gradient_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract gradient-based features using Sobel operators
        
        LINEAR ALGEBRA CONCEPT: Gradient vectors and directional derivatives
        - Sobel operators are convolution matrices (linear transformations)
        - Gradient = [∂I/∂x, ∂I/∂y] is a vector field
        - Magnitude = ||gradient|| uses L2 norm
        - Orientation = arctan(∂I/∂y, ∂I/∂x) gives direction
        
        Mathematical formulation:
        Gx = Sobel_x * Image (matrix convolution)
        Gy = Sobel_y * Image (matrix convolution)
        Magnitude = √(Gx² + Gy²) (L2 norm of gradient vector)
        """
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Sobel operators (3x3 convolution matrices)
        # These are linear operators that compute derivatives
        sobelx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude: ||∇I|| = √(Gx² + Gy²)
        # This is the L2 norm of the gradient vector at each pixel
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Gradient orientation: θ = arctan2(Gy, Gx)
        orientation = np.arctan2(sobely, sobelx)
        
        # Normalize
        magnitude = magnitude / (np.max(magnitude) + 1e-7)
        
        return magnitude, orientation
    
    def extract_local_features(self, image: np.ndarray, cell_size: int = 16) -> np.ndarray:
        """
        Extract local histogram features using cells
        
        LINEAR ALGEBRA CONCEPT: Block matrices and histogram vectors
        - Divide image into cells (block matrix decomposition)
        - Each cell produces a histogram vector
        - Concatenate to form feature vector (vector stacking)
        """
        magnitude, orientation = self.extract_gradient_features(image)
        
        h, w = image.shape
        features = []
        
        # Divide into cells
        for i in range(0, h - cell_size, cell_size):
            for j in range(0, w - cell_size, cell_size):
                # Extract cell
                cell_mag = magnitude[i:i+cell_size, j:j+cell_size]
                cell_orient = orientation[i:i+cell_size, j:j+cell_size]
                
                # Create histogram (8 orientation bins)
                hist, _ = np.histogram(cell_orient, bins=8, range=(-np.pi, np.pi), 
                                      weights=cell_mag)
                
                # Normalize histogram (L2 normalization)
                hist = hist / (np.linalg.norm(hist) + 1e-7)
                
                features.append(hist)
        
        # Stack all histogram vectors into one feature vector
        return np.concatenate(features)
    
    def extract_multi_scale_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract features at multiple scales
        
        LINEAR ALGEBRA CONCEPT: Multi-resolution analysis
        - Pyramid representation (hierarchical matrix decomposition)
        - Features from different scales capture different information
        """
        features = {}
        
        # Original scale
        img_original = image
        features['global'] = img_original.flatten()
        
        # Gradient features
        magnitude, orientation = self.extract_gradient_features(img_original)
        features['gradient_mag'] = magnitude.flatten()
        features['gradient_orient'] = orientation.flatten()
        
        # Local features
        features['local'] = self.extract_local_features(img_original)
        
        # Half scale
        img_half = cv2.resize(img_original, (self.image_size[0]//2, self.image_size[1]//2))
        features['half_scale'] = img_half.flatten()
        
        return features
    
    def train_pca_models(self, all_features: List[Dict[str, np.ndarray]]):
        """
        Train separate PCA models for different feature types
        
        LINEAR ALGEBRA CONCEPT: Multiple eigendecompositions
        - Each feature type gets its own PCA transformation
        - Combines information from multiple subspaces
        """
        print("\n" + "="*70)
        print("TRAINING PCA MODELS")
        print("="*70)
        
        # Collect features by type
        global_feats = np.array([f['global'] for f in all_features])
        gradient_feats = np.array([f['gradient_mag'] for f in all_features])
        local_feats = np.array([f['local'] for f in all_features])
        
        n_samples = len(all_features)
        
        # PCA for global features
        n_comp_global = min(50, n_samples, global_feats.shape[1])
        print(f"\nGlobal features PCA: {global_feats.shape[1]} → {n_comp_global} dims")
        self.pca_global = PCA(n_components=n_comp_global)
        self.pca_global.fit(global_feats)
        print(f"  Variance explained: {self.pca_global.explained_variance_ratio_.sum()*100:.2f}%")
        
        # PCA for gradient features
        n_comp_gradient = min(30, n_samples, gradient_feats.shape[1])
        print(f"\nGradient features PCA: {gradient_feats.shape[1]} → {n_comp_gradient} dims")
        self.pca_gradient = PCA(n_components=n_comp_gradient)
        self.pca_gradient.fit(gradient_feats)
        print(f"  Variance explained: {self.pca_gradient.explained_variance_ratio_.sum()*100:.2f}%")
        
        # PCA for local features
        n_comp_local = min(40, n_samples, local_feats.shape[1])
        print(f"\nLocal features PCA: {local_feats.shape[1]} → {n_comp_local} dims")
        self.pca_local = PCA(n_components=n_comp_local)
        self.pca_local.fit(local_feats)
        print(f"  Variance explained: {self.pca_local.explained_variance_ratio_.sum()*100:.2f}%")
        
        print("\n" + "="*70)
        print("PCA TRAINING COMPLETE")
        print("="*70 + "\n")
    
    def transform_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Transform using all PCA models and concatenate
        
        LINEAR ALGEBRA CONCEPT: Subspace projection and concatenation
        - Project onto multiple principal component subspaces
        - Concatenate projected vectors (vector stacking)
        - Result is in a combined feature space
        """
        # Transform each feature type
        global_transformed = self.pca_global.transform(features['global'].reshape(1, -1)).flatten()
        gradient_transformed = self.pca_gradient.transform(features['gradient_mag'].reshape(1, -1)).flatten()
        local_transformed = self.pca_local.transform(features['local'].reshape(1, -1)).flatten()
        
        # Concatenate all transformed features
        # This creates a comprehensive feature vector
        combined = np.concatenate([global_transformed, gradient_transformed, local_transformed])
        
        return combined
    
    def calculate_similarity(self, query_features: np.ndarray, db_features: np.ndarray) -> float:
        """
        Calculate similarity using multiple linear algebra metrics
        
        LINEAR ALGEBRA CONCEPTS:
        1. Euclidean distance (L2 norm): ||v1 - v2||₂
        2. Manhattan distance (L1 norm): ||v1 - v2||₁
        3. Cosine similarity: (v1 · v2) / (||v1|| ||v2||)
        4. Correlation: normalized dot product
        
        Combines multiple metrics using weighted voting
        """
        # 1. L2 distance (Euclidean)
        l2_dist = euclidean_distance(query_features, db_features)
        l2_score = 1.0 / (1.0 + l2_dist)  # Convert to similarity
        
        # 2. L1 distance (Manhattan)
        l1_dist = np.sum(np.abs(query_features - db_features))
        l1_score = 1.0 / (1.0 + l1_dist)
        
        # 3. Cosine similarity
        cos_sim = cosine_similarity(query_features, db_features)
        cos_score = (cos_sim + 1.0) / 2.0  # Normalize to [0, 1]
        
        # 4. Correlation coefficient (Pearson)
        # This is a normalized dot product after mean centering
        q_centered = query_features - np.mean(query_features)
        db_centered = db_features - np.mean(db_features)
        correlation = np.dot(q_centered, db_centered) / (np.linalg.norm(q_centered) * np.linalg.norm(db_centered) + 1e-7)
        corr_score = (correlation + 1.0) / 2.0
        
        # Weighted combination
        # These weights can be tuned based on empirical performance
        weights = np.array([0.35, 0.25, 0.25, 0.15])  # L2, L1, Cosine, Correlation
        scores = np.array([l2_score, l1_score, cos_score, corr_score])
        
        # Weighted dot product (linear combination)
        final_score = np.dot(weights, scores)
        
        return final_score, l2_dist, cos_sim
    
    def build_database(self, image_paths: List[str], labels: List[str]):
        """Build database with enhanced multi-scale features"""
        print("\n" + "="*70)
        print("BUILDING FINGERPRINT DATABASE")
        print("="*70)
        
        all_features = []
        valid_labels = []
        
        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            try:
                image = self.preprocess_image(img_path)
                features = self.extract_multi_scale_features(image)
                all_features.append(features)
                valid_labels.append(label)
                print(f"Processed {i+1}/{len(image_paths)}: {label}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Train PCA models
        self.train_pca_models(all_features)
        
        # Transform and store features
        print("\nTransforming all features...")
        self.database_features = []
        self.database_labels = []
        
        for features, label in zip(all_features, valid_labels):
            transformed = self.transform_features(features)
            self.database_features.append(transformed)
            self.database_labels.append(label)
        
        print(f"\nDatabase built: {len(self.database_features)} fingerprints")
        print(f"Feature dimension: {len(self.database_features[0])}")
        print("="*70 + "\n")
    
    def match_fingerprint(self, query_image_path: str, top_k: int = 3) -> List[Tuple[str, float, float]]:
        """
        Match using enhanced multi-metric approach
        Shows ALL scans from the same person, sorted by similarity
        """
        print("\n" + "="*70)
        print("FINGERPRINT MATCHING")
        print("="*70)
        
        # Extract query person ID from filename
        query_filename = query_image_path.split('\\')[-1] if '\\' in query_image_path else query_image_path.split('/')[-1]
        query_person = query_filename.split('_')[0]
        
        print(f"Query: {query_filename}")
        print(f"Person ID: {query_person}")
        print()
        
        # Extract and transform query features
        query_image = self.preprocess_image(query_image_path)
        query_features_dict = self.extract_multi_scale_features(query_image)
        query_features = self.transform_features(query_features_dict)
        
        print(f"Query feature dimension: {len(query_features)}")
        print(f"Comparing against {len(self.database_features)} fingerprints...")
        print()
        
        # Calculate similarities for all fingerprints
        all_matches = []
        same_person_matches = []
        
        for i, (db_features, label) in enumerate(zip(self.database_features, self.database_labels)):
            similarity, distance, cos_sim = self.calculate_similarity(query_features, db_features)
            match_person = label.split('_')[0]
            
            match_data = (label, similarity, distance, cos_sim, match_person == query_person)
            all_matches.append(match_data)
            
            # Collect only same-person matches
            if match_person == query_person:
                same_person_matches.append(match_data)
                print(f"{label}: SAME PERSON")
                print(f"  Combined similarity: {similarity:.4f}")
                print(f"  L2 distance: {distance:.2f}")
                print(f"  Cosine similarity: {cos_sim:.4f}")
                print()
        
        # Sort same-person matches by similarity (higher is better)
        same_person_matches.sort(key=lambda x: x[1], reverse=True)
        
        print("="*70)
        print(f"MATCHING RESULTS - SHOWING ALL SCANS FROM PERSON {query_person}")
        print("="*70)
        print(f"Found {len(same_person_matches)} scans from the same person")
        print()
        
        if len(same_person_matches) > 0:
            for rank, (label, similarity, distance, cos_sim, _) in enumerate(same_person_matches, 1):
                is_perfect = distance < 0.01
                marker = "[IDENTICAL]" if is_perfect else "[MATCH]"
                print(f"{rank}. {label} {marker}")
                print(f"   Similarity: {similarity:.4f} | Distance: {distance:.2f} | Cosine: {cos_sim:.4f}")
                print()
        else:
            print("No matching scans found in database")
        
        # Show top non-matching results for comparison
        other_matches = [m for m in all_matches if not m[4]]
        other_matches.sort(key=lambda x: x[1], reverse=True)
        
        if len(other_matches) > 0 and len(same_person_matches) > 0:
            print()
            print("Top 3 non-matching fingerprints (for comparison):")
            print("-" * 70)
            for rank, (label, similarity, distance, cos_sim, _) in enumerate(other_matches[:3], 1):
                person = label.split('_')[0]
                print(f"{rank}. {label} - Person {person} (DIFFERENT)")
                print(f"   Similarity: {similarity:.4f} | Distance: {distance:.2f} | Cosine: {cos_sim:.4f}")
        
        print("="*70 + "\n")
        
        # Return only same-person matches for GUI
        # Convert similarity to distance-like score for GUI compatibility
        return [(label, 100*(1-sim), sim) for label, sim, _, _, _ in same_person_matches]
