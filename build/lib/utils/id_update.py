#!/usr/bin/env python3
import sqlite3
import argparse
from pathlib import Path

def get_animal_columns(conn):
    """Identify animal-related columns while excluding detection table"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT m.name AS table_name, p.name AS column_name 
        FROM sqlite_master AS m
        JOIN pragma_table_info(m.name) AS p
        WHERE (p.name LIKE 'idanimal%' 
               OR p.name = 'mouse_id'
               OR (m.name = 'ANIMAL' AND p.name = 'ID'))
          AND m.name != 'DETECTION'
    """)
    return cursor.fetchall()

def update_mouse_id(db_path, old_id, new_id, dry_run=False):
    """Safely update mouse ID with proper resource management"""
    conn = None
    try:
        # Initialize connection
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = OFF")
        cursor = conn.cursor()

        # Validate input IDs
        cursor.execute("SELECT ID FROM ANIMAL WHERE ID = ?", (old_id,))
        if not cursor.fetchone():
            raise ValueError(f"Old ID {old_id} not found in ANIMAL table")
        
        cursor.execute("SELECT ID FROM ANIMAL WHERE ID = ?", (new_id,))
        if cursor.fetchone():
            raise ValueError(f"New ID {new_id} already exists in ANIMAL table")

        # Get target columns
        columns = get_animal_columns(conn)
        if not columns:
            raise ValueError("No animal-related columns found")

        changes = []
        with conn:
            # Update dependent tables first
            for table, column in columns:
                if table == "ANIMAL" and column == "ID":
                    continue  # Handle ANIMAL last
                
                query = f"UPDATE {table} SET {column} = ? WHERE {column} = ?"
                params = (new_id, old_id)
                
                if dry_run:
                    print(f"[DRY RUN] Would execute: {query} {params}")
                    continue
                
                cursor.execute(query, params)
                changes.append((table, column, cursor.rowcount))

            # Update ANIMAL table last
            if not dry_run:
                cursor.execute(
                    "UPDATE ANIMAL SET ID = ? WHERE ID = ?",
                    (new_id, old_id)
                )
                changes.append(("ANIMAL", "ID", cursor.rowcount))

        # Post-update verification
        if not dry_run:
            print("\nUpdate summary:")
            for table, column, count in changes:
                print(f"- {table}.{column}: {count} rows updated")
            
            # Foreign key check
            cursor.execute("PRAGMA foreign_key_check")
            if errors := cursor.fetchall():
                raise RuntimeError(f"Foreign key issues detected: {errors}")

        print(f"\n✅ {'Dry run completed' if dry_run else 'Update successful'}")

    except sqlite3.Error as e:
        print(f"❌ Database error: {str(e)}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        if conn:
            try:
                # Reset foreign keys and close connection
                conn.execute("PRAGMA foreign_keys = ON")
                conn.close()
                print("✅ Database connection closed properly")
            except sqlite3.Error as e:
                print(f"⚠️ Warning: Error during cleanup - {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Safely update mouse IDs in SQLite database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("database", help="Path to SQLite database file")
    parser.add_argument("old_id", type=int, help="Current mouse ID to replace")
    parser.add_argument("new_id", type=int, help="New unique mouse ID")
    parser.add_argument("--dry-run", action="store_true",
                      help="Simulate changes without modifying database")
    
    args = parser.parse_args()
    
    print(f"Database: {Path(args.database).resolve()}")
    print(f"Changing ID: {args.old_id} → {args.new_id}")
    
    if args.dry_run:
        print("\n🚧 DRY RUN MODE - No changes will be written")
    
    confirm = input("\n❗ Confirm update (y/N): ").lower()
    if confirm == 'y':
        update_mouse_id(args.database, args.old_id, args.new_id, args.dry_run)
    else:
        print("\n🚫 Operation cancelled")

    print("\nID REPLACED SUCCESSFULLY!") 