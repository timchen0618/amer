#!/bin/bash

# Script to add "hypersearch_" prefix to all directory names in current folder

# Check if any directories exist
if ! ls -d */ >/dev/null 2>&1; then
    echo "No directories found in the current folder."
    exit 0
fi

echo "Adding 'hypersearch_' prefix to all directories in current folder..."

# Loop through all directories in current folder
for dir in */; do
    # Remove trailing slash from directory name
    dir_name="${dir%/}"
    
    # Skip if directory already has the prefix
    if [[ "$dir_name" == hypersearch_* ]]; then
        echo "Skipping '$dir_name' - already has prefix"
        continue
    fi
    
    # Create new name with prefix
    new_name="hypersearch_$dir_name"
    
    # Check if target directory already exists
    if [ -d "$new_name" ]; then
        echo "Warning: '$new_name' already exists. Skipping '$dir_name'"
        continue
    fi
    
    # Rename the directory
    mv "$dir_name" "$new_name"
    
    if [ $? -eq 0 ]; then
        echo "Renamed: '$dir_name' -> '$new_name'"
    else
        echo "Error: Failed to rename '$dir_name'"
    fi
done

echo "Done!" 