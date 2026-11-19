import os
import shutil
from pathlib import Path


def remove_cache_files(root_directory=".", dry_run=True):
    """
    Recursively traverse the file system and delete files that start with 'cache'.
    
    Args:
        root_directory (str): The root directory to start searching from. Defaults to current directory.
        dry_run (bool): If True, only print what would be deleted without actually deleting. Defaults to True.
    
    Returns:
        list: List of files that were (or would be) deleted
    """
    deleted_files = []
    root_path = Path(root_directory).resolve()
    
    print(f"Searching for cache files in: {root_path}")
    print(f"Dry run mode: {'ON' if dry_run else 'OFF'}")
    print("-" * 50)
    
    try:
        # Walk through all files in the directory tree
        for file_path in root_path.rglob("*"):
            if file_path.is_file() and file_path.name.lower().startswith("cache"):
                deleted_files.append(str(file_path))
                
                if dry_run:
                    print(f"[DRY RUN] Would delete: {file_path}")
                else:
                    try:
                        file_path.unlink()  # Delete the file
                        print(f"[DELETED] {file_path}")
                    except PermissionError:
                        print(f"[ERROR] Permission denied: {file_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to delete {file_path}: {e}")
    
    except Exception as e:
        print(f"Error traversing directory: {e}")
        return deleted_files
    
    print("-" * 50)
    print(f"Total files {'found' if dry_run else 'deleted'}: {len(deleted_files)}")
    
    return deleted_files


def remove_cache_files_interactive(root_directory="."):
    """
    Interactive version that asks for confirmation before deleting files.
    
    Args:
        root_directory (str): The root directory to start searching from.
    
    Returns:
        list: List of files that were deleted
    """
    # First, do a dry run to see what would be deleted
    print("Scanning for cache files...")
    cache_files = remove_cache_files(root_directory, dry_run=True)
    
    if not cache_files:
        print("No cache files found.")
        return []
    
    # Ask for confirmation
    print(f"\nFound {len(cache_files)} cache files.")
    response = input("Do you want to delete these files? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        print("\nDeleting files...")
        return remove_cache_files(root_directory, dry_run=False)
    else:
        print("Deletion cancelled.")
        return []


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Usage: python remove_cache_files.py [directory] [--dry-run] [--interactive]")
            print("  directory: Root directory to search (default: current directory)")
            print("  --dry-run: Only show what would be deleted (default)")
            print("  --interactive: Ask for confirmation before deleting")
            print("  --force: Actually delete files without confirmation")
            sys.exit(0)
        
        directory = sys.argv[1] if not sys.argv[1].startswith("--") else "."
        dry_run = "--dry-run" in sys.argv or "--force" not in sys.argv
        interactive = "--interactive" in sys.argv
        
        if interactive:
            remove_cache_files_interactive(directory)
        else:
            remove_cache_files(directory, dry_run=dry_run)
    else:
        # Default behavior: dry run in current directory
        remove_cache_files(".", dry_run=True)
