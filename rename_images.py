import os
import sys

def rename_image_files(directory):
    """
    Rename all files in the specified directory by appending '_folder_2' 
    before the file extension.
    
    Args:
        directory (str): Path to the directory containing image files
    """
    # Check if directory exists
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Get the full file path
        filepath = os.path.join(directory, filename)
        
        # Check if it's a file (not a subdirectory)
        if os.path.isfile(filepath):
            # Split the filename and extension
            name, ext = os.path.splitext(filename)
            
            # Create new filename
            new_filename = f"{name}_folder_6{ext}"
            new_filepath = os.path.join(directory, new_filename)
            
            # Rename the file
            try:
                os.rename(filepath, new_filepath)
                print(f"Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

def main():
    # Check if directory path is provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python rename_images.py /path/to/image/folder")
        sys.exit(1)
    
    # Get directory path from command line argument
    directory = sys.argv[1]
    
    # Rename files
    rename_image_files(directory)

if __name__ == "__main__":
    main()

# Example usage:
# python rename_images.py /path/to/your/image/folder