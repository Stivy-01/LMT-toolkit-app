"""SQLite Table Manager - Simple and efficient database operations"""

import sqlite3, tkinter as tk
from tkinter import messagebox, ttk, simpledialog
from pathlib import Path
import sys
import threading

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.database_utils import get_table_columns
from src.utils.db_selector import get_db_path

def enhance_table(db_path, table_name):
    """Add SEX, AGE, STRAIN, SETUP columns to table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create new table with additional columns
        cursor.execute(f"CREATE TABLE {table_name}_new AS SELECT * FROM {table_name}")
        
        # Add new columns
        for col in ['SEX', 'AGE', 'STRAIN', 'SETUP']:
            try:
                cursor.execute(f"ALTER TABLE {table_name}_new ADD COLUMN {col} TEXT")
            except:
                pass  # Column might already exist
            
        # Replace old table
        cursor.execute(f"DROP TABLE {table_name}")
        cursor.execute(f"ALTER TABLE {table_name}_new RENAME TO {table_name}")
        conn.commit()
        
        return True, f"Added columns to {table_name}"
    except Exception as e:
        conn.rollback()
        return False, f"Error: {str(e)}"
    finally:
        conn.close()

def remove_columns(db_path, table_name, columns_to_keep):
    """Remove columns from a table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create new table with only selected columns
        cols_str = ", ".join(columns_to_keep)
        cursor.execute(f"CREATE TABLE {table_name}_new AS SELECT {cols_str} FROM {table_name}")
        
        # Replace old table
        cursor.execute(f"DROP TABLE {table_name}")
        cursor.execute(f"ALTER TABLE {table_name}_new RENAME TO {table_name}")
        conn.commit()
        
        return True, f"Removed columns from {table_name}"
    except Exception as e:
        conn.rollback()
        return False, f"Error: {str(e)}"
    finally:
        conn.close()

def delete_rows(db_path, table_name, id_column, id_value):
    """Delete rows from a table by ID"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Delete rows matching the ID
        cursor.execute(f"DELETE FROM {table_name} WHERE {id_column} = ?", (id_value,))
        rows_affected = cursor.rowcount
        conn.commit()
        
        return True, f"Deleted {rows_affected} rows from {table_name}"
    except Exception as e:
        conn.rollback()
        return False, f"Error: {str(e)}"
    finally:
        conn.close()

class SQLiteTableManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SQLite Table Manager")
        self.root.geometry("550x400")
        self.db_path = None
        self.selected_table = None
        self.setup_ui()
    
    def setup_ui(self):
        # Database selection
        db_frame = tk.Frame(self.root, padx=10, pady=10)
        db_frame.pack(fill=tk.X)
        
        tk.Button(db_frame, text="Select Database", command=self.select_db).pack(side=tk.LEFT, padx=5)
        self.db_label = tk.Label(db_frame, text="No database selected")
        self.db_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Table selection
        table_frame = tk.Frame(self.root, padx=10, pady=5)
        table_frame.pack(fill=tk.X)
        
        tk.Label(table_frame, text="Table:").pack(side=tk.LEFT)
        self.table_var = tk.StringVar()
        self.table_menu = tk.OptionMenu(table_frame, self.table_var, "")
        self.table_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.table_var.trace("w", lambda *args: self.on_table_selected())
        
        # Create notebook with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add Columns Tab
        self.add_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.add_tab, text="Add Columns")
        self.setup_add_columns_tab()
        
        # Remove Columns Tab
        self.remove_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.remove_tab, text="Remove Columns")
        self.setup_remove_columns_tab()
        
        # Delete Rows Tab
        self.delete_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.delete_tab, text="Delete Rows")
        self.setup_delete_rows_tab()
        
        # Progress indicator
        self.status_frame = tk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress = ttk.Progressbar(self.status_frame, mode="indeterminate", length=200)
        self.progress.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
    
    def setup_add_columns_tab(self):
        # Simple button to add columns
        tk.Label(self.add_tab, text="Add SEX, AGE, STRAIN, SETUP columns to the selected table").pack(pady=20)
        tk.Button(self.add_tab, text="Add Columns", command=self.enhance_table).pack(pady=10)
    
    def setup_remove_columns_tab(self):
        # Columns selection
        tk.Label(self.remove_tab, text="Select columns to keep:").pack(anchor=tk.W, pady=5)
        
        list_frame = tk.Frame(self.remove_tab)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.columns_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set)
        self.columns_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.columns_listbox.yview)
        
        # Buttons
        btn_frame = tk.Frame(self.remove_tab)
        btn_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(btn_frame, text="Select All", command=lambda: self.columns_listbox.select_set(0, tk.END)).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Deselect All", command=lambda: self.columns_listbox.selection_clear(0, tk.END)).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Remove Columns", command=self.remove_columns).pack(side=tk.RIGHT, padx=5)
    
    def setup_delete_rows_tab(self):
        # ID column selection
        id_frame = tk.Frame(self.delete_tab)
        id_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(id_frame, text="ID Column:").pack(side=tk.LEFT, padx=5)
        self.id_column_var = tk.StringVar()
        self.id_column_menu = tk.OptionMenu(id_frame, self.id_column_var, "")
        self.id_column_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Delete button
        tk.Button(self.delete_tab, text="Delete Rows by ID", command=self.delete_rows).pack(pady=10)
        
        # Instructions
        tk.Label(self.delete_tab, text="This will prompt you to enter an ID value.\nAll rows matching that ID will be deleted.").pack(pady=20)
    
    def start_progress(self, message="Processing..."):
        """Start the progress indicator"""
        self.status_label.config(text=message)
        self.progress.start(10)
    
    def stop_progress(self, message="Complete"):
        """Stop the progress indicator"""
        self.progress.stop()
        self.status_label.config(text=message)
    
    def select_db(self):
        paths = get_db_path()
        if paths and len(paths) > 0:
            self.db_path = paths[0]
            self.db_label.config(text=f"Database: {Path(self.db_path).name}")
            self.start_progress("Loading tables...")
            threading.Thread(target=self._load_tables_thread).start()
    
    def _load_tables_thread(self):
        """Load tables in a separate thread"""
        if not self.db_path:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Update UI in main thread
        self.root.after(0, lambda: self._update_tables_ui(tables))
    
    def _update_tables_ui(self, tables):
        """Update UI with tables (called from main thread)"""
        menu = self.table_menu["menu"]
        menu.delete(0, tk.END)
        for table in tables:
            menu.add_command(label=table, command=lambda t=table: self.table_var.set(t))
        
        if tables:
            self.table_var.set(tables[0])
        
        self.stop_progress("Tables loaded")
    
    def on_table_selected(self):
        self.selected_table = self.table_var.get()
        if self.selected_table:
            self.start_progress("Loading columns...")
            threading.Thread(target=self._load_columns_thread).start()
    
    def _load_columns_thread(self):
        """Load columns in a separate thread"""
        if not self.db_path or not self.selected_table:
            return
            
        conn = sqlite3.connect(self.db_path)
        columns = get_table_columns(conn, self.selected_table)
        conn.close()
        
        # Update UI in main thread
        self.root.after(0, lambda: self._update_columns_ui(columns))
    
    def _update_columns_ui(self, columns):
        """Update UI with columns (called from main thread)"""
        # Update columns listbox
        self.columns_listbox.delete(0, tk.END)
        for col in columns:
            self.columns_listbox.insert(tk.END, col)
        self.columns_listbox.select_set(0, tk.END)
        
        # Update ID column dropdown
        menu = self.id_column_menu["menu"]
        menu.delete(0, tk.END)
        for col in columns:
            menu.add_command(label=col, command=lambda c=col: self.id_column_var.set(c))
        
        if columns:
            # Try to find an ID column, or use the first column
            id_cols = [col for col in columns if col.upper() in ('ID', 'ROWID', 'MOUSE_ID', 'ANIMAL_ID')]
            if id_cols:
                self.id_column_var.set(id_cols[0])
            else:
                self.id_column_var.set(columns[0])
        
        self.stop_progress("Columns loaded")
    
    def enhance_table(self):
        if not self.db_path or not self.selected_table:
            messagebox.showwarning("Warning", "Select a database and table first")
            return
            
        if messagebox.askyesno("Confirm", f"Add SEX, AGE, STRAIN, SETUP columns to table '{self.selected_table}'?"):
            self.start_progress("Adding columns...")
            threading.Thread(target=self._enhance_table_thread).start()
    
    def _enhance_table_thread(self):
        """Run enhance table in a separate thread"""
        try:
           
            # Enhance table
            success, msg = enhance_table(self.db_path, self.selected_table)
            
            # Update UI in main thread
            if success:
                self.root.after(0, lambda: [
                    messagebox.showinfo("Success", msg),
                    self.on_table_selected(),
                    self.stop_progress("Columns added")
                ])
            else:
                self.root.after(0, lambda: [
                    messagebox.showinfo("Info", msg),
                    self.stop_progress("Operation failed")
                ])
        except Exception as e:
            self.root.after(0, lambda: [
                messagebox.showerror("Error", f"Operation failed: {str(e)}"),
                self.stop_progress("Error")
            ])
    
    def remove_columns(self):
        if not self.db_path or not self.selected_table:
            messagebox.showwarning("Warning", "Select a database and table first")
            return
            
        selected_indices = self.columns_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Select at least one column to keep")
            return
        
        columns_to_keep = [self.columns_listbox.get(i) for i in selected_indices]
        
        if messagebox.askyesno("Confirm", f"Remove all columns except:\n{', '.join(columns_to_keep)}\n\nThis cannot be undone!"):
            self.start_progress("Removing columns...")
            threading.Thread(target=lambda: self._remove_columns_thread(columns_to_keep)).start()
    
    def _remove_columns_thread(self, columns_to_keep):
        """Run remove columns in a separate thread"""
        try:
      
            # Remove columns
            success, msg = remove_columns(self.db_path, self.selected_table, columns_to_keep)
            
            # Update UI in main thread
            if success:
                self.root.after(0, lambda: [
                    messagebox.showinfo("Success", msg),
                    self.on_table_selected(),
                    self.stop_progress("Columns removed")
                ])
            else:
                self.root.after(0, lambda: [
                    messagebox.showerror("Error", msg),
                    self.stop_progress("Operation failed")
                ])
        except Exception as e:
            self.root.after(0, lambda: [
                messagebox.showerror("Error", f"Operation failed: {str(e)}"),
                self.stop_progress("Error")
            ])
    
    def delete_rows(self):
        if not self.db_path or not self.selected_table:
            messagebox.showwarning("Warning", "Select a database and table first")
            return
            
        id_column = self.id_column_var.get()
        if not id_column:
            messagebox.showwarning("Warning", "Select an ID column")
            return
        
        # Prompt for ID value
        id_value = simpledialog.askstring("Input", f"Enter {id_column} value to delete:")
        if not id_value:
            return
        
        if messagebox.askyesno("Confirm", f"Delete all rows where {id_column} = '{id_value}'?\n\nThis cannot be undone!"):
            self.start_progress("Deleting rows...")
            threading.Thread(target=lambda: self._delete_rows_thread(id_column, id_value)).start()
    
    def _delete_rows_thread(self, id_column, id_value):
        """Run delete rows in a separate thread"""
        try:
            
            # Delete rows
            success, msg = delete_rows(self.db_path, self.selected_table, id_column, id_value)
            
            # Update UI in main thread
            if success:
                self.root.after(0, lambda: [
                    messagebox.showinfo("Success", msg),
                    self.stop_progress("Rows deleted")
                ])
            else:
                self.root.after(0, lambda: [
                    messagebox.showerror("Error", msg),
                    self.stop_progress("Operation failed")
                ])
        except Exception as e:
            self.root.after(0, lambda: [
                messagebox.showerror("Error", f"Operation failed: {str(e)}"),
                self.stop_progress("Error")
            ])

def main():
    root = tk.Tk()
    SQLiteTableManagerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 