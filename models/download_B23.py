import os
import zipfile
import shutil

# Main directories
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root project directory
DATABASE_DIR = os.path.join(PROJECT_DIR, "BD")  # BD folder inside the project

# URL of the POLLEN23E database
DATABASE_URL = "https://figshare.com/ndownloader/articles/1525086/versions/1"

def ask_user(prompt):
    """
    Asks the user whether to proceed with an action.
    Returns True if the user responds 'y' (yes), False otherwise.
    """
    response = input(f"{prompt} (y/n): ").strip().lower()
    return response == "y"

def download_database(url, output_file):
    """
    Downloads the ZIP file using wget, if the user agrees.
    """
    if os.path.exists(output_file):
        print(f"The file '{output_file}' already exists.")
        if not ask_user("Do you want to download it again?"):
            return

    try:
        import subprocess
        print(f"Downloading the file '{output_file}'...")
        subprocess.run(["wget", "-O", output_file, url], check=True)
        print(f"Download completed: '{output_file}'.")
    except Exception as e:
        print(f"Error downloading the file: {e}")
        exit(1)

def extract_and_rename_zip(zip_path, target_dir, new_name):
    """
    Extracts the ZIP file content into a temporary folder, renames the folder, and organizes the images,
    if the user agrees.
    """
    extracted_dir = os.path.join(target_dir, new_name)
    temp_dir = os.path.join(target_dir, "temp_extract")

    if os.path.exists(extracted_dir) or os.path.exists(temp_dir):
        print(f"The folder '{new_name}' or '{temp_dir}' already exists.")
        if not ask_user("Do you want to extract it again?"):
            return extracted_dir if os.path.exists(extracted_dir) else temp_dir

    try:
        # Create the temporary folder
        os.makedirs(temp_dir, exist_ok=True)

        # Extract the ZIP file content into the temporary folder
        print(f"Extracting the file '{zip_path}' into '{temp_dir}'...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print(f"Extraction completed in '{temp_dir}'.")

        # Rename the temporary folder to the new name
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)  # Remove the folder if it already exists
        os.rename(temp_dir, extracted_dir)
        print(f"Folder renamed to '{new_name}'.")

        return extracted_dir
    except Exception as e:
        print(f"Error during extraction or renaming: {e}")
        exit(1)

def organize_images_by_class(base_dir):
    """
    Organizes images into subfolders based on class names extracted from filenames.
    """
    print(f"Organizing images in '{base_dir}' by class...")
    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)

        # Process only files (ignore folders)
        if os.path.isfile(file_path):
            # Remove the file extension
            class_name = os.path.splitext(filename)[0]

            # Remove numbers, underscores, and special characters after the main name
            class_name = ''.join([char for char in class_name if not char.isdigit() and char != '_']).strip()

            # Remove parentheses and extra spaces
            class_name = class_name.split('(')[0].strip()

            if not class_name:  # If the class name is invalid
                print(f"Error extracting class name from file: {filename}")
                continue

            # Create a subfolder for the class, if it doesn't exist
            class_dir = os.path.join(base_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Move the file to the corresponding subfolder
            new_file_path = os.path.join(class_dir, filename)
            shutil.move(file_path, new_file_path)
            print(f"File '{filename}' moved to folder '{class_name}'.")

    print("Organization completed.")

def clean_compressed_file(zip_file):
    """
    Asks the user whether to delete the compressed file after extraction.
    """
    if os.path.exists(zip_file):
        if ask_user(f"Do you want to delete the compressed file '{zip_file}'?"):
            try:
                os.remove(zip_file)
                print(f"Compressed file '{zip_file}' deleted successfully.")
            except Exception as e:
                print(f"Error deleting the compressed file: {e}")
        else:
            print("The compressed file was not deleted.")

def main():
    # Define paths for the ZIP file and target directory
    os.makedirs(DATABASE_DIR, exist_ok=True)  # Ensure the BD folder exists
    zip_file = os.path.join(DATABASE_DIR, "1525086.zip")
    renamed_dir_name = "POLLEN23E"

    # Step 1: Download the ZIP file
    download_database(DATABASE_URL, zip_file)

    # Step 2: Extract the ZIP file and rename the folder
    extracted_dir = extract_and_rename_zip(zip_file, DATABASE_DIR, renamed_dir_name)

    # Step 3: Organize images into folders by class
    organize_images_by_class(extracted_dir)

    # Step 4: Clean up the compressed file (optional)
    clean_compressed_file(zip_file)

    print("Processing completed!")

if __name__ == "__main__":
    main()