#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import re
import glob
from datetime import datetime, timedelta
from tkcalendar import Calendar

def get_db_path():
    """GUI-based database file selection with support for multiple files"""
    root = tk.Tk()
    root.withdraw()
    
    # Create custom dialog
    choice = messagebox.askquestion(
        "Selection Mode",
        "How would you like to select files?",
        icon='question',
        detail="Select 'Yes' for folder processing, 'No' for manual file selection (you can select multiple files using Ctrl/Cmd)"
    )
    
    if choice == 'yes':
        # Folder selection with recursive SQLite file search
        folder = filedialog.askdirectory(title="Select Folder Containing SQLite Files")
        if not folder:
            return []
            
        files = []
        for path in glob.glob(f"{folder}/**/*.sqlite", recursive=True):
            if Path(path).is_file():
                files.append(str(Path(path).resolve()))
        return files
    
    # Multiple file selection
    files = filedialog.askopenfilenames(
        title="Select SQLite Database File(s) - Use Ctrl/Cmd for multiple selection",
        filetypes=[("SQLite Databases", "*.sqlite"), ("All Files", "*.*")],
        multiple=True  # Explicitly enable multiple selection
    )
    
    if not files:
        return []
        
    return [str(Path(f).resolve()) for f in files]

def get_date_from_filename(db_path):
    """Extract date from filename."""
    path = Path(db_path)
    # Try different date formats
    patterns = [
        r"(\d{2})[\._](\d{2})[\._](\d{2})",  # dd.mm.yy or dd_mm_yy
    ]
    
    for pattern in patterns:
        match = re.search(pattern, path.name)
        if match:
            return f"20{match.group(3)}-{match.group(2)}-{match.group(1)}"
    
    raise ValueError(f"No valid date found in filename: {path.name}")

def get_experiment_time():
    """GUI dialog to select both date and time"""
    root = tk.Tk()
    root.withdraw()
    
    try:
        # Date selection
        top = tk.Toplevel(root)
        top.title("Select Experiment Start Time")
        cal = Calendar(top, date_pattern='y-mm-dd')
        cal.pack(padx=10, pady=10)

        # Time selection
        time_frame = tk.Frame(top)
        time_frame.pack(pady=5)
        
        tk.Label(time_frame, text="Time (HH:MM:SS) - 24h format").pack(side=tk.LEFT)
        
        # Create entry widgets for direct input
        hour_var = tk.StringVar(value="16")  # Set initial value
        min_var = tk.StringVar(value="30")   # Set initial value
        sec_var = tk.StringVar(value="00")   # Set initial value
        
        def create_entry(frame, var, width=2):
            entry = tk.Entry(frame, textvariable=var, width=width, justify=tk.CENTER)
            entry.bind('<FocusIn>', lambda e: entry.selection_range(0, tk.END))
            return entry
        
        hour_entry = create_entry(time_frame, hour_var)
        hour_entry.pack(side=tk.LEFT)
        hour_entry.insert(0, hour_var.get())  # Explicitly set the text
        
        tk.Label(time_frame, text=":").pack(side=tk.LEFT)
        min_entry = create_entry(time_frame, min_var)
        min_entry.pack(side=tk.LEFT)
        min_entry.insert(0, min_var.get())  # Explicitly set the text
        
        tk.Label(time_frame, text=":").pack(side=tk.LEFT)
        sec_entry = create_entry(time_frame, sec_var)
        sec_entry.pack(side=tk.LEFT)
        sec_entry.insert(0, sec_var.get())  # Explicitly set the text

        # Confirmation
        selected_time = None
        def on_confirm():
            nonlocal selected_time
            try:
                date_str = cal.get_date()
                
                # Get the actual values from the entries
                hour = int(hour_entry.get())
                minute = int(min_entry.get())
                second = int(sec_entry.get())
                
                # Validate ranges
                if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
                    raise ValueError("Time values out of range")
                
                time_str = f"{hour:02d}:{minute:02d}:{second:02d}"
                print(f"Final time string: {time_str}")
                selected_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                top.destroy()
            except ValueError as e:
                print(f"Error details: {str(e)}")  # Print the actual error
                messagebox.showerror("Error", f"Invalid time values. Please check your input.\nHour: {hour_entry.get()}\nMinute: {min_entry.get()}\nSecond: {sec_entry.get()}")

        tk.Button(top, text="Confirm", command=on_confirm).pack(pady=5)
        root.wait_window(top)
        return selected_time

    except ImportError:
        # Fallback to text input
        root.destroy()
        while True:
            time_str = input("Enter experiment start (YYYY-MM-DD HH:MM:SS) - 24h format: ")
            try:
                return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                print("Invalid format! Please use YYYY-MM-DD HH:MM:SS in 24h format") 