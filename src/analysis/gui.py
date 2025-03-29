"""
GUI Module
=========

Provides graphical user interface components for:
- Dataset selection
- Analysis type configuration
- Parameter settings
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import pandas as pd
import threading
from queue import Queue
import sys

class AnalysisGUI:
    """
    GUI for configuring and running behavioral analysis.
    """
    
    def __init__(self):
        """Initialize the GUI."""
        self.root = tk.Tk()
        self.root.title("LMT Analysis Configuration")
        self.root.geometry("600x400")
        
        # Style configuration
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('Info.TLabel', font=('Helvetica', 9, 'italic'))
        
        # Create message queue for thread communication
        self.msg_queue = Queue()
        
        self.create_widgets()
        
        # Start message checking
        self.check_msg_queue()
        
    def create_widgets(self):
        """Create and arrange GUI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title = ttk.Label(
            main_frame, 
            text="LMT Behavioral Analysis Configuration",
            style='Title.TLabel'
        )
        title.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 1. Dataset Selection
        dataset_frame = ttk.LabelFrame(main_frame, text="Dataset Selection", padding="5")
        dataset_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.dataset_path = tk.StringVar()
        ttk.Label(dataset_frame, text="Dataset:").grid(row=0, column=0, padx=5)
        ttk.Entry(dataset_frame, textvariable=self.dataset_path, width=50).grid(
            row=0, column=1, padx=5
        )
        ttk.Button(dataset_frame, text="Browse...", command=self.browse_dataset).grid(
            row=0, column=2, padx=5
        )
        
        # 2. Analysis Configuration
        config_frame = ttk.LabelFrame(main_frame, text="Analysis Configuration", padding="5")
        config_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Analysis Type
        ttk.Label(config_frame, text="Analysis Type:").grid(row=0, column=0, padx=5, pady=5)
        self.analysis_type = tk.StringVar(value="both")
        ttk.Radiobutton(
            config_frame, text="Sequential Only", 
            variable=self.analysis_type, value="sequential"
        ).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(
            config_frame, text="Parallel Only", 
            variable=self.analysis_type, value="parallel"
        ).grid(row=0, column=2, padx=5)
        ttk.Radiobutton(
            config_frame, text="Both Approaches", 
            variable=self.analysis_type, value="both"
        ).grid(row=0, column=3, padx=5)
        
        # Parameters Section
        param_frame = ttk.LabelFrame(config_frame, text="Analysis Parameters", padding="5")
        param_frame.grid(row=1, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E))
        
        # Parameter description
        ttk.Label(
            param_frame,
            text="These are the established parameters used in the analysis:",
            style='Info.TLabel'
        ).grid(row=0, column=0, columnspan=4, pady=(0, 5), padx=5)
        
        # Correlation Threshold (read-only)
        ttk.Label(param_frame, text="Correlation Threshold:").grid(
            row=1, column=0, padx=5, pady=2
        )
        self.corr_threshold = tk.StringVar(value="0.95")
        ttk.Entry(
            param_frame,
            textvariable=self.corr_threshold,
            width=10,
            state='readonly'
        ).grid(row=1, column=1, padx=5)
        
        # Variance Threshold (read-only)
        ttk.Label(param_frame, text="Variance Threshold:").grid(
            row=1, column=2, padx=5, pady=2
        )
        self.var_threshold = tk.StringVar(value="0.1")
        ttk.Entry(
            param_frame,
            textvariable=self.var_threshold,
            width=10,
            state='readonly'
        ).grid(row=1, column=3, padx=5)
        
        # Parameter explanations
        ttk.Label(
            param_frame,
            text="Correlation: Features with correlation > 0.95 are merged",
            style='Info.TLabel'
        ).grid(row=2, column=0, columnspan=2, pady=(5,0), padx=5, sticky=tk.W)
        
        ttk.Label(
            param_frame,
            text="Variance: Features with variance < 0.1 are removed",
            style='Info.TLabel'
        ).grid(row=2, column=2, columnspan=2, pady=(5,0), padx=5, sticky=tk.W)
        
        # 3. Output Configuration
        output_frame = ttk.LabelFrame(main_frame, text="Output Configuration", padding="5")
        output_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.output_path = tk.StringVar()
        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, padx=5)
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).grid(
            row=0, column=1, padx=5
        )
        ttk.Button(output_frame, text="Browse...", command=self.browse_output).grid(
            row=0, column=2, padx=5
        )
        
        # Run Button
        self.run_button = ttk.Button(
            main_frame, 
            text="Run Analysis",
            command=self.run_analysis,
            style='Accent.TButton'
        )
        self.run_button.grid(row=4, column=0, columnspan=2, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            length=580
        )
        self.progress.grid(row=5, column=0, columnspan=2, pady=(0, 10))
        
        # Status
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(
            main_frame, 
            textvariable=self.status_var,
            wraplength=580
        )
        self.status_label.grid(row=6, column=0, columnspan=2)
        
    def browse_dataset(self):
        """Open file dialog for dataset selection."""
        filename = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.dataset_path.set(filename)
            
            # Auto-set output directory based on dataset location
            default_output = Path(filename).parent / "analyzed"
            self.output_path.set(str(default_output))
    
    def browse_output(self):
        """Open directory dialog for output location."""
        directory = filedialog.askdirectory(
            title="Select Output Directory"
        )
        if directory:
            self.output_path.set(directory)
    
    def validate_inputs(self):
        """Validate all user inputs."""
        # Check dataset path
        if not self.dataset_path.get():
            messagebox.showerror("Error", "Please select a dataset file.")
            return False
        
        if not Path(self.dataset_path.get()).exists():
            messagebox.showerror("Error", "Selected dataset file does not exist.")
            return False
        
        # Check output path
        if not self.output_path.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return False
        
        return True
    
    def check_msg_queue(self):
        """Check for messages from the analysis thread."""
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg.startswith("Status:"):
                    self.status_var.set(msg[7:])
                elif msg == "Done":
                    self.progress.stop()
                    self.progress.grid_remove()
                    self.run_button.config(state='normal')
                    messagebox.showinfo("Complete", "Analysis completed successfully!")
                elif msg.startswith("Error:"):
                    self.progress.stop()
                    self.progress.grid_remove()
                    self.run_button.config(state='normal')
                    messagebox.showerror("Error", msg[6:])
        except:
            pass
        finally:
            # Schedule next check
            self.root.after(100, self.check_msg_queue)
    
    def run_analysis_thread(self, config):
        """Run analysis in a separate thread."""
        try:
            from analysis.analysis_pipeline import AnalysisPipeline
            
            # Create output directory if it doesn't exist
            output_path = Path(config['output_dir'])
            output_path.mkdir(parents=True, exist_ok=True)
            
            def progress_callback(msg):
                self.msg_queue.put(msg)
            
            # Initialize and run pipeline
            pipeline = AnalysisPipeline(output_dir=output_path)
            pipeline.run_analysis(
                data_path=config['dataset_path'],
                analysis_type=config['analysis_type'],
                correlation_threshold=0.95,  # Using default value
                variance_threshold=0.1,      # Using default value
                progress_callback=progress_callback
            )
            
            self.msg_queue.put("Done")
            
        except Exception as e:
            self.msg_queue.put(f"Error: {str(e)}")
    
    def run_analysis(self):
        """Validate inputs and run the analysis in a separate thread."""
        if not self.validate_inputs():
            return
        
        # Get configuration
        config = {
            'dataset_path': self.dataset_path.get(),
            'output_dir': str(Path(self.output_path.get())),
            'analysis_type': self.analysis_type.get()
        }
        
        # Disable run button and show progress
        self.run_button.config(state='disabled')
        self.progress.grid()
        self.progress.start(10)
        self.status_var.set("Starting analysis...")
        
        # Start analysis thread
        thread = threading.Thread(target=self.run_analysis_thread, args=(config,))
        thread.daemon = True
        thread.start()
    
    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()

def get_analysis_config():
    """
    Show GUI to get analysis configuration from user.
    
    Returns:
        dict: Analysis configuration if user completes setup,
              None if user cancels
    """
    gui = AnalysisGUI()
    return gui.run() 