"""
Simple GUI for Fingerprint Recognition System using Tkinter
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
from typing import List, Tuple

from enhanced_processor import EnhancedFingerprintProcessor
from utils import get_image_files


class FingerprintGUI:
    """
    Simple Tkinter-based GUI for fingerprint matching
    """
    
    def __init__(self, root: tk.Tk, processor: EnhancedFingerprintProcessor, 
                 query_images: List[str], database_images: List[str]):
        """
        Initialize the GUI
        
        Args:
            root: Tkinter root window
            processor: EnhancedFingerprintProcessor instance
            query_images: List of query image paths
            database_images: List of database image paths
        """
        self.root = root
        self.processor = processor
        self.query_images = query_images
        self.database_images = database_images
        
        # Configure window
        self.root.title("Fingerprint Recognition System - Linear Algebra Demo")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2b2b2b')
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'), 
                           background='#2b2b2b', foreground='#ffffff')
        self.style.configure('Subtitle.TLabel', font=('Arial', 10), 
                           background='#2b2b2b', foreground='#cccccc')
        self.style.configure('Result.TLabel', font=('Arial', 11), 
                           background='#3c3c3c', foreground='#ffffff', 
                           padding=10, relief='raised')
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Title
        title_frame = tk.Frame(self.root, bg='#2b2b2b', pady=20)
        title_frame.pack(fill='x')
        
        title = ttk.Label(title_frame, text="Fingerprint Recognition System", 
                         style='Title.TLabel')
        title.pack()
        
        subtitle = ttk.Label(title_frame, 
                           text="Demonstrating Linear Algebra in Biometric Authentication", 
                           style='Subtitle.TLabel')
        subtitle.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Query selection
        left_frame = tk.Frame(main_frame, bg='#3c3c3c', relief='raised', bd=2)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        query_label = tk.Label(left_frame, text="Select Query Fingerprint", 
                              font=('Arial', 12, 'bold'), bg='#3c3c3c', fg='#ffffff')
        query_label.pack(pady=10)
        
        # Query selection dropdown
        self.query_var = tk.StringVar()
        query_names = [os.path.basename(path) for path in self.query_images]
        
        if query_names:
            self.query_var.set(query_names[0])
        
        query_dropdown = ttk.Combobox(left_frame, textvariable=self.query_var, 
                                     values=query_names, state='readonly', width=30)
        query_dropdown.pack(pady=10)
        query_dropdown.bind('<<ComboboxSelected>>', self.on_query_selected)
        
        # Query image display
        self.query_image_label = tk.Label(left_frame, bg='#3c3c3c', 
                                         text="No image selected", 
                                         fg='#888888', width=30, height=15)
        self.query_image_label.pack(pady=10)
        
        # Match button
        match_btn = tk.Button(left_frame, text="Match Fingerprint", 
                            command=self.match_fingerprint,
                            font=('Arial', 12, 'bold'), bg='#4CAF50', fg='white',
                            activebackground='#45a049', cursor='hand2',
                            relief='raised', bd=3, padx=20, pady=10)
        match_btn.pack(pady=20)
        
        # Info label
        info_text = ("This system uses Principal Component Analysis (PCA)\n"
                    "to reduce dimensionality and Euclidean distance\n"
                    "to find matching fingerprints.")
        info_label = tk.Label(left_frame, text=info_text, 
                            font=('Arial', 9), bg='#3c3c3c', fg='#cccccc',
                            justify='center')
        info_label.pack(side='bottom', pady=10)
        
        # Right panel - Results
        right_frame = tk.Frame(main_frame, bg='#3c3c3c', relief='raised', bd=2)
        right_frame.pack(side='right', fill='both', expand=True)
        
        results_label = tk.Label(right_frame, text="Matching Results", 
                               font=('Arial', 12, 'bold'), bg='#3c3c3c', fg='#ffffff')
        results_label.pack(pady=10)
        
        # Results display area with scrollbar
        canvas_frame = tk.Frame(right_frame, bg='#3c3c3c')
        canvas_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(canvas_frame, bg='#3c3c3c', highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        
        self.results_frame = tk.Frame(canvas, bg='#3c3c3c')
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        canvas_window = canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        
        # Configure canvas scrolling
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        self.results_frame.bind("<Configure>", configure_scroll_region)
        
        # Update canvas window width
        def configure_canvas_width(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind("<Configure>", configure_canvas_width)
        
        # Initial message
        initial_msg = tk.Label(self.results_frame, 
                             text="Select a query fingerprint and click 'Match Fingerprint'", 
                             font=('Arial', 10), bg='#3c3c3c', fg='#888888',
                             wraplength=400)
        initial_msg.pack(expand=True)
        
        # Load initial query image
        if query_names:
            self.on_query_selected(None)
    
    def on_query_selected(self, event):
        """Handle query image selection"""
        query_name = self.query_var.get()
        query_path = next((path for path in self.query_images 
                         if os.path.basename(path) == query_name), None)
        
        if query_path:
            self.display_image(query_path, self.query_image_label, size=(200, 200))
    
    def display_image(self, image_path: str, label: tk.Label, size: Tuple[int, int] = (150, 150)):
        """Display an image in a label"""
        try:
            img = Image.open(image_path)
            img = img.resize(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label.configure(image=photo, text="")
            label.image = photo  # Keep a reference
        except Exception as e:
            label.configure(text=f"Error loading image:\n{str(e)}", 
                          fg='#ff6b6b')
    
    def match_fingerprint(self):
        """Perform fingerprint matching"""
        query_name = self.query_var.get()
        
        if not query_name:
            messagebox.showwarning("No Selection", "Please select a query fingerprint")
            return
        
        query_path = next((path for path in self.query_images 
                         if os.path.basename(path) == query_name), None)
        
        if not query_path:
            messagebox.showerror("Error", "Query image not found")
            return
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Show processing message
        processing_label = tk.Label(self.results_frame, 
                                   text="Processing... Please wait", 
                                   font=('Arial', 11), bg='#3c3c3c', fg='#ffd700')
        processing_label.pack(expand=True)
        self.root.update()
        
        try:
            # Perform matching
            matches = self.processor.match_fingerprint(query_path, top_k=3)
            
            # Clear processing message
            processing_label.destroy()
            
            # Display results
            self.display_results(matches)
            
        except Exception as e:
            processing_label.destroy()
            error_label = tk.Label(self.results_frame, 
                                 text=f"Error during matching:\n{str(e)}", 
                                 font=('Arial', 10), bg='#3c3c3c', fg='#ff6b6b',
                                 wraplength=400)
            error_label.pack(expand=True)
    
    def display_results(self, matches: List[Tuple[str, float, float]]):
        """Display matching results - showing all scans from the same person"""
        
        # Extract person ID from query filename
        query_name = self.query_var.get()
        query_person_id = query_name.split('_')[0]  # Get '101' from '101_5.tif'
        
        if len(matches) == 0:
            error_label = tk.Label(self.results_frame, 
                                 text="No matching fingerprints found in database!", 
                                 font=('Arial', 11), bg='#3c3c3c', fg='#ff6b6b',
                                 wraplength=400)
            error_label.pack(expand=True)
            return
        
        # Header
        header_text = f"All Scans from Person {query_person_id} ({len(matches)} found)"
        header = tk.Label(self.results_frame, text=header_text,
                         font=('Arial', 11, 'bold'), bg='#3c3c3c', fg='#4CAF50',
                         pady=10)
        header.pack(fill='x')
        
        for i, (label, distance, similarity) in enumerate(matches, 1):
            # Extract person ID from match
            match_person_id = label.split('_')[0]
            is_same_person = (match_person_id == query_person_id)
            is_identical = distance < 1  # Very low distance = same scan
            
            # Create frame for each result
            result_frame = tk.Frame(self.results_frame, bg='#4a4a4a', 
                                   relief='raised', bd=2)
            result_frame.pack(fill='x', pady=3, padx=5)
            
            # Rank with special marker for identical scan
            if is_identical:
                rank_text = "*"
                rank_color = '#FFD700'
            else:
                rank_text = f"#{i}"
                rank_color = '#4CAF50'
            
            rank_label = tk.Label(result_frame, text=rank_text, 
                                font=('Arial', 12, 'bold'), bg='#4a4a4a', 
                                fg=rank_color, width=3)
            rank_label.pack(side='left', padx=5)
            
            # Image - smaller for multiple results
            db_path = next((path for path in self.database_images 
                          if os.path.basename(path) == label), None)
            
            if db_path:
                img_label = tk.Label(result_frame, bg='#4a4a4a')
                img_label.pack(side='left', padx=3, pady=3)
                self.display_image(db_path, img_label, size=(60, 60))
            
            # Info
            info_frame = tk.Frame(result_frame, bg='#4a4a4a')
            info_frame.pack(side='left', fill='both', expand=True, padx=5)
            
            name_label = tk.Label(info_frame, text=label, 
                                font=('Arial', 9, 'bold'), bg='#4a4a4a', 
                                fg='#ffffff', anchor='w')
            name_label.pack(fill='x')
            
            # Show if it's identical scan or different scan
            if is_identical:
                status_text = "[IDENTICAL]"
                status_color = '#FFD700'
            else:
                status_text = "MATCH"
                status_color = '#4CAF50'
            
            status_label = tk.Label(info_frame, text=status_text, 
                                  font=('Arial', 8, 'bold'), bg='#4a4a4a', 
                                  fg=status_color, anchor='w')
            status_label.pack(fill='x')
            
            # Similarity score
            similarity_label = tk.Label(info_frame, 
                                      text=f"Similarity: {similarity:.4f}", 
                                      font=('Arial', 8), bg='#4a4a4a', 
                                      fg='#cccccc', anchor='w')
            similarity_label.pack(fill='x')
        
        # Add summary at bottom
        summary_frame = tk.Frame(self.results_frame, bg='#3c3c3c')
        summary_frame.pack(fill='x', pady=15)
        
        summary_text = (f"Successfully matched {len(matches)} scans\n"
                       f"All results are from Person {query_person_id}\n"
                       f"Using Multi-Scale PCA and Gradient Analysis")
        
        summary_label = tk.Label(summary_frame, 
                               text=summary_text, 
                               font=('Arial', 8), bg='#3c3c3c', fg='#888888',
                               justify='center')
        summary_label.pack()
