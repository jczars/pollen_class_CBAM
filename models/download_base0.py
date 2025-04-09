import os
import subprocess
import argparse
import re
import cv2
from tqdm import tqdm

# Define the project directory and database directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Navigate up to the project root
DATABASE_DIR = os.path.join(PROJECT_DIR, "BD")  # BD folder inside the project

# Dictionary with URLs of available databases
DATABASES = {
    "CPD1_Cr": "https://zenodo.org/records/4756361/files/Cropped%20Pollen%20Grains.rar?download=1",  # Example URL for CPD1
    # Add more databases here if needed
}

def file_exists(directory, filename):
    """Checks if a file or directory already exists in the specified directory."""
    return os.path.exists(os.path.join(directory, filename))

def is_renamed(directory):
    """Checks if the classes in the directory have been renamed."""
    for item in os.listdir(directory):
        if re.match(r"^\d+\.([A-Za-z]+)$", item):  # Check if any class matches the old naming pattern
            return False
    return True

def download_with_wget(url, output_file):
    """Downloads the file using wget."""
    try:
        subprocess.run(["wget", "-O", output_file, url], check=True)
        print(f"Download completed successfully. File saved as '{output_file}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading with wget: {e}")
        exit(1)

def extract_rar(file_path, output_dir):
    """Extracts the RAR file using the unrar command."""
    try:
        # Ensure the 'unrar' command is available
        subprocess.run(["unrar", "x", file_path, output_dir], check=True)
        print(f"File extracted successfully into '{output_dir}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting the file: {e}")
        print("Ensure the 'unrar' command is installed on your system.")
        exit(1)

def get_extracted_directory(output_dir):
    """
    Detects the name of the extracted directory dynamically.
    Assumes there is only one main directory created after extraction.
    """
    try:
        # List all items in the output directory
        items = os.listdir(output_dir)
        # Filter out only directories
        directories = [item for item in items if os.path.isdir(os.path.join(output_dir, item))]
        if not directories:
            raise FileNotFoundError(f"No directories found in '{output_dir}' after extraction.")
        # Return the first directory (assuming only one main directory is created)
        return os.path.join(output_dir, directories[0])
    except Exception as e:
        print(f"Error detecting extracted directory: {e}")
        exit(1)

def rename_extracted_directory(extracted_dir, new_name):
    """
    Renames the extracted directory to the desired name.
    """
    try:
        new_path = os.path.join(os.path.dirname(extracted_dir), new_name)
        os.rename(extracted_dir, new_path)
        print(f"Renamed extracted directory to: '{new_name}'")
        return new_path
    except Exception as e:
        print(f"Error renaming extracted directory: {e}")
        exit(1)

def rename_classes(base_dir):
    """
    Renames classes in the directory to remove numbers and capitalize correctly.
    Navigates into subdirectories if necessary.
    """
    print("Renaming classes...")
    
    # Rename items in the target directory
    renamed_count = 0
    for item in os.listdir(base_dir):
        old_path = os.path.join(base_dir, item)
        if os.path.isdir(old_path):  # Check if it's a directory
            # Use regex to identify the pattern "1.Thymbra" and rename to "thymbra"
            match = re.match(r"^\d+\.([A-Za-z]+)$", item)
            if match:
                new_name = match.group(1).lower()  # Convert to lowercase
                new_path = os.path.join(base_dir, new_name)
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: '{item}' -> '{new_name}'")
                    renamed_count += 1
                except Exception as e:
                    print(f"Error renaming '{item}': {e}")
    
    if renamed_count == 0:
        print("No classes were found to rename. Ensure the directory structure and naming patterns are correct.")

def clean_compressed_files(output_dir, compressed_file):
    """Asks the user if they want to delete the compressed file and deletes it if confirmed."""
    response = input("Do you want to delete the compressed file after extraction? (yes/no): ").strip().lower()
    if response == "yes":
        try:
            os.remove(compressed_file)
            print(f"Compressed file '{compressed_file}' deleted successfully.")
        except Exception as e:
            print(f"Error deleting the compressed file: {e}")
    else:
        print("Compressed file was not deleted.")

def resize_images_in_place(src, target_size=(224, 224), file_type='png', verbose=0):
    """
    Verifies if images in the source directory have different sizes and resizes them to the target size (224x224).
    The resized images are saved in the same directory, overwriting the original files.
    """
    # Check if the source directory exists
    if not os.path.exists(src):
        print(f"Error: The directory '{src}' does not exist.")
        return

    if verbose > 0:
        print(f"Processing images in directory: {src}")

    # Iterate over subdirectories in the source directory
    for subdir in tqdm(os.listdir(src), desc="Processing subdirectories"):
        subdir_path = os.path.join(src, subdir)
        
        # Skip if it's not a directory
        if not os.path.isdir(subdir_path):
            if verbose > 0:
                print(f"Skipping non-directory: {subdir}")
            continue

        if verbose > 0:
            print(f"\nProcessing subdirectory: {subdir}")

        # Iterate over files in the subdirectory
        for filename in os.listdir(subdir_path):
            if filename.endswith(f".{file_type}"):
                img_path = os.path.join(subdir_path, filename)

                try:
                    # Load the image
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Error: Unable to load image '{img_path}'. Skipping...")
                        continue

                    # Get the current dimensions of the image
                    height, width = img.shape[:2]

                    # Check if the image dimensions are different from the target size
                    if (width, height) != target_size:
                        if verbose > 0:
                            print(f"Image '{filename}' has size ({width}, {height}). Resizing to {target_size}...")

                        # Resize the image
                        img_resized = cv2.resize(img, target_size)

                        # Save the resized image back to the same path (overwrite the original)
                        cv2.imwrite(img_path, img_resized)
                        if verbose > 0:
                            print(f"Resized and overwritten: '{img_path}'")
                    else:
                        if verbose > 0:
                            print(f"Image '{filename}' already has size {target_size}. No resizing needed.")

                except Exception as e:
                    print(f"Error processing image '{filename}': {e}")

    if verbose > 0:
        print("\nProcessing completed.")

def main():
    # Argument parser configuration
    parser = argparse.ArgumentParser(description="Script to download, extract, rename, and resize datasets.")
    parser.add_argument(
        "-d", "--database", choices=list(DATABASES.keys()), required=True,
        help="Name of the database to download. Available options: " + ", ".join(DATABASES.keys())
    )
    args = parser.parse_args()

    # Get the URL of the selected database
    database_url = DATABASES.get(args.database)
    if not database_url:
        print(f"Error: Database '{args.database}' not found.")
        exit(1)

    # Define the output file name and target directory
    os.makedirs(DATABASE_DIR, exist_ok=True)  # Create the directory if it doesn't exist
    output_file = os.path.join(DATABASE_DIR, f"{args.database}.rar")

    # Check if the database directory already exists
    if file_exists(DATABASE_DIR, args.database):
        print(f"The database '{args.database}' already exists in '{DATABASE_DIR}'.")
        response = input("Do you want to overwrite it? (yes/no): ").strip().lower()
        if response != "yes":
            # Check if the classes are already renamed
            if is_renamed(os.path.join(DATABASE_DIR, args.database)):
                print("The classes are already renamed. Proceeding to resize images...")
                resize_images_in_place(os.path.join(DATABASE_DIR, args.database), verbose=1)
                exit(0)
            else:
                response_rename = input("The classes are not renamed. Do you want to rename them now? (yes/no): ").strip().lower()
                if response_rename == "yes":
                    rename_classes(os.path.join(DATABASE_DIR, args.database))
                    resize_images_in_place(os.path.join(DATABASE_DIR, args.database), verbose=1)
                    exit(0)
                else:
                    print("No changes were made.")
                    exit(0)

    # Perform the download
    print(f"Downloading the database '{args.database}'...")
    download_with_wget(database_url, output_file)

    # Extract the file
    print(f"Extracting the database '{args.database}'...")
    extract_rar(output_file, DATABASE_DIR)

    # Detect the extracted directory dynamically
    extracted_dir = get_extracted_directory(DATABASE_DIR)
    print(f"Detected extracted directory: '{extracted_dir}'")

    # Rename the extracted directory to the desired name
    renamed_dir = rename_extracted_directory(extracted_dir, args.database)
    print(f"Using renamed directory: '{renamed_dir}'")

    # Rename the classes
    rename_classes(renamed_dir)

    # Resize the images
    print("Resizing images to 224x224...")
    resize_images_in_place(renamed_dir, verbose=1)

    # Clean up the compressed file if requested
    clean_compressed_files(DATABASE_DIR, output_file)

if __name__ == "__main__":
    main()