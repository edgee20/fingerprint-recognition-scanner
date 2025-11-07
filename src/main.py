"""
Main entry point for the Fingerprint Recognition System

This application demonstrates linear algebra concepts in biometric authentication:
- Matrix operations (image representation)
- Principal Component Analysis (eigendecomposition)
- Vector projections and transformations
- Distance metrics (L2 norm)
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox

from enhanced_processor import EnhancedFingerprintProcessor
from gui import FingerprintGUI
from utils import get_image_files, create_sample_fingerprint


def create_sample_dataset():
    """Create sample fingerprint images if dataset doesn't exist"""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Define dataset paths
    db_dir = os.path.join(project_dir, 'dataset', 'db')
    query_dir = os.path.join(project_dir, 'dataset', 'query')
    
    # Create directories if they don't exist
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)
    
    # Check if we already have images
    db_images = get_image_files(db_dir)
    query_images = get_image_files(query_dir)
    
    if len(db_images) >= 10 and len(query_images) >= 3:
        print("Found existing dataset with realistic fingerprints!")
        return db_dir, query_dir
    
    if len(db_images) >= 5 and len(query_images) >= 2:
        print("Found existing dataset!")
        return db_dir, query_dir
    
    print("Creating sample fingerprint dataset...")
    print("Note: For realistic fingerprints, run: python setup_real_dataset.py")
    print()
    
    # Create sample database fingerprints
    import cv2
    import numpy as np
    
    for i in range(1, 6):
        output_path = os.path.join(db_dir, f'fingerprint_db_{i:02d}.png')
        if not os.path.exists(output_path):
            # Create unique synthetic patterns
            img = np.zeros((96, 96), dtype=np.uint8)
            
            # Add ridge-like patterns with variation
            offset = i * 5
            for j in range(offset, 96, 3 + (i % 2)):
                cv2.line(img, (j, 0), (j, 96), 200, 1)
            
            # Add some circular patterns
            cv2.circle(img, (48 + i*2, 48 - i*2), 20 + i*3, 150, 1)
            
            # Add noise for uniqueness
            noise = np.random.randint(0, 30 + i*10, (96, 96), dtype=np.uint8)
            img = cv2.add(img, noise)
            
            cv2.imwrite(output_path, img)
            print(f"Created: {os.path.basename(output_path)}")
    
    # Create sample query fingerprints (similar to some database ones)
    for i in range(1, 3):
        output_path = os.path.join(query_dir, f'fingerprint_query_{i}.png')
        if not os.path.exists(output_path):
            # Make query similar to a database fingerprint
            base_idx = i * 2
            img = np.zeros((96, 96), dtype=np.uint8)
            
            offset = base_idx * 5
            for j in range(offset, 96, 3 + (base_idx % 2)):
                cv2.line(img, (j, 0), (j, 96), 200, 1)
            
            cv2.circle(img, (48 + base_idx*2, 48 - base_idx*2), 20 + base_idx*3, 150, 1)
            
            # Different noise to simulate same finger, different scan
            noise = np.random.randint(0, 40, (96, 96), dtype=np.uint8)
            img = cv2.add(img, noise)
            
            cv2.imwrite(output_path, img)
            print(f"Created: {os.path.basename(output_path)}")
    
    print("\nSample dataset created successfully!")
    print(f"Database: {db_dir}")
    print(f"Query: {query_dir}")
    print()
    
    return db_dir, query_dir


def main():
    """Main function to run the fingerprint recognition system"""
    
    print("="*70)
    print("FINGERPRINT RECOGNITION SYSTEM")
    print("Linear Algebra in Biometric Authentication")
    print("="*70)
    print()
    
    # Create or locate dataset
    try:
        db_dir, query_dir = create_sample_dataset()
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
    
    # Get image files
    db_images = get_image_files(db_dir)
    query_images = get_image_files(query_dir)
    
    if len(db_images) == 0:
        print("Error: No database images found!")
        print(f"Please add fingerprint images to: {db_dir}")
        return
    
    if len(query_images) == 0:
        print("Error: No query images found!")
        print(f"Please add fingerprint images to: {query_dir}")
        return
    
    print(f"Found {len(db_images)} database fingerprints")
    print(f"Found {len(query_images)} query fingerprints")
    print()
    
    # Initialize processor with enhanced algorithm
    print("Initializing Fingerprint Processor...")
    processor = EnhancedFingerprintProcessor(n_components=100, image_size=(256, 256))
    
    # Build database
    try:
        db_labels = [os.path.basename(path) for path in db_images]
        processor.build_database(db_images, db_labels)
    except Exception as e:
        print(f"Error building database: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Launch GUI
    print("Launching GUI...")
    print()
    print("="*70)
    print("INSTRUCTIONS:")
    print("1. Select a query fingerprint from the dropdown")
    print("2. Click 'Match Fingerprint' to find similar prints")
    print("3. View all matches from the same person")
    print("="*70)
    print()
    
    try:
        root = tk.Tk()
        app = FingerprintGUI(root, processor, query_images, db_images)
        root.mainloop()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nThank you for using the Fingerprint Recognition System!")


if __name__ == "__main__":
    main()
