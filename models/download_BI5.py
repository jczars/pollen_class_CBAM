import os
import subprocess
import argparse
import cv2
from tqdm import tqdm
import zipfile

import sound_test_finished

# Define the project directory and database directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Navigate up to the project root
DATABASE_DIR = os.path.join(PROJECT_DIR, "BD")  # BD folder inside the project

# Dictionary with URLs of available databases
DATABASES = {
    "BI_Cr_5": "https://zenodo.org/records/14188979/files/BI_Cr_5.zip?download=1",  # Example URL for BI5
    # Add more databases here if needed
}

def file_exists(directory, filename):
    """Checks if a file or directory already exists in the specified directory."""
    return os.path.exists(os.path.join(directory, filename))

def is_valid_zip(file_path):
    """Checks if the file is a valid ZIP archive."""
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            return True
    except zipfile.BadZipFile:
        return False

def download_with_wget(url, output_file):
    """Downloads the file using wget."""
    try:
        subprocess.run(["wget", "-O", output_file, url], check=True)
        print(f"Download completed successfully. File saved as '{output_file}'.")
        
        # Verify if the downloaded file is a valid ZIP
        if not is_valid_zip(output_file):
            print(f"The file '{output_file}' is not a valid ZIP archive. Please check the URL.")
            exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading with wget: {e}")
        exit(1)

def extract_zip(file_path, output_dir):
    """Extracts the ZIP file using the unzip command."""
    try:
        # Ensure the 'unzip' command is available
        subprocess.run(["unzip", file_path, "-d", output_dir], check=True)
        print(f"File extracted successfully into '{output_dir}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting the file: {e}")
        print("Ensure the 'unzip' command is installed on your system.")
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

def resize_images_in_place(src, target_size=(224, 224), file_types=['png', 'jpg', 'jpeg'], verbose=0):
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
            if any(filename.endswith(f".{file_type}") for file_type in file_types):
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

def main():
    # Argument parser configuration
    parser = argparse.ArgumentParser(description="Script to download, extract, and resize datasets.")
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
    output_file = os.path.join(DATABASE_DIR, f"{args.database}.zip")
    extracted_dir = os.path.join(DATABASE_DIR, args.database)

    # Step 1: Download the database (if not already downloaded)
    if not file_exists(DATABASE_DIR, f"{args.database}.zip"):
        print(f"Downloading the database '{args.database}'...")
        download_with_wget(database_url, output_file)
    else:
        print(f"The database '{args.database}' is already downloaded.")

    # Step 2: Extract the database (if not already extracted)
    if not file_exists(DATABASE_DIR, args.database):
        print(f"Extracting the database '{args.database}'...")
        extract_zip(output_file, DATABASE_DIR)

        # Detect the extracted directory dynamically
        extracted_dir = get_extracted_directory(DATABASE_DIR)
        print(f"Detected extracted directory: '{extracted_dir}'")
    else:
        print(f"The database '{args.database}' is already extracted.")
        extracted_dir = os.path.join(DATABASE_DIR, args.database)

    # Step 3: Resize images (if not already resized)
    print("Resizing images to 224x224...")
    resize_images_in_place(extracted_dir, verbose=1)

    # Step 4: Clean up the compressed file (optional)
    clean_compressed_files(DATABASE_DIR, output_file)
    sound_test_finished.beep(2)

if __name__ == "__main__":
    main()