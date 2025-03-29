# -*- coding: utf-8 -*-
import os
from pathlib import Path

def fix_file_encoding(file_path):
    """Fix file encoding to UTF-8 without BOM"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                if content is not None:
                    break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print(f"❌ Could not read {file_path} with any encoding")
            return False
        
        # Write back with UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        print(f"✅ Fixed encoding for {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {str(e)}")
        return False

def main():
    # Get the project root (one level up from tools directory)
    project_root = Path(__file__).parent.parent
    
    # Find all Python files
    python_files = []
    for root, _, files in os.walk(project_root / 'src'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    print(f"Found {len(python_files)} Python files")
    
    # Fix encoding for each file
    success = 0
    for file_path in python_files:
        if fix_file_encoding(file_path):
            success += 1
    
    print(f"\nFixed {success} out of {len(python_files)} files")

if __name__ == "__main__":
    main() 