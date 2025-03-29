#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog
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
    choice = tk.messagebox.askquestion(
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
    """Extract date from filename (modified for Path objects)"""
    path = Path(db_path)
    match = re.search(r"\b(\d{2})\.(\d{2})\.(\d{2})\b", path.name)
    if not match:
        raise ValueError(f"No valid date found in filename: {path.name}")
    return f"20{match.group(3)}-{match.group(2)}-{match.group(1)}"

def get_experiment_time():
    """GUI dialog to select both date and time"""
    root = tk.Tk()
    root.withdraw()
    
    try:
        # Date selection
        from tkcalendar import Calendar
        top = tk.Toplevel(root)
        cal = Calendar(top, date_pattern='y-mm-dd')
        cal.pack(padx=10, pady=10)

        # Time selection
        time_frame = tk.Frame(top)
        time_frame.pack(pady=5)
        
        tk.Label(time_frame, text="Time (HH:MM:SS)").pack(side=tk.LEFT)
        hour_var = tk.StringVar(value='09')
        min_var = tk.StringVar(value='00')
        sec_var = tk.StringVar(value='00')
        
        tk.Spinbox(time_frame, from_=0, to=23, width=2, textvariable=hour_var).pack(side=tk.LEFT)
        tk.Label(time_frame, text=":").pack(side=tk.LEFT)
        tk.Spinbox(time_frame, from_=0, to=59, width=2, textvariable=min_var).pack(side=tk.LEFT)
        tk.Label(time_frame, text=":").pack(side=tk.LEFT)
        tk.Spinbox(time_frame, from_=0, to=59, width=2, textvariable=sec_var).pack(side=tk.LEFT)

        # Confirmation
        selected_time = None
        def on_confirm():
            nonlocal selected_time
            date_str = cal.get_date()
            time_str = f"{hour_var.get().zfill(2)}:{min_var.get().zfill(2)}:{sec_var.get().zfill(2)}"
            selected_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            top.destroy()

        tk.Button(top, text="Confirm", command=on_confirm).pack(pady=5)
        root.wait_window(top)
        return selected_time

    except ImportError:
        # Fallback to text input
        root.destroy()
        while True:
            time_str = input("Enter experiment start (YYYY-MM-DD HH:MM:SS): ")
            try:
                return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                print("Invalid format! Please use YYYY-MM-DD HH:MM:SS") 