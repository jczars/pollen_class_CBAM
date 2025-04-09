import os
import zipfile
import shutil

# Main directories
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root project directory
DATABASE_DIR = os.path.join(PROJECT_DIR, "BD")  # BD folder inside the project

# URL of the POLLEN73S database
DATABASE_URL = "https://figshare.com/ndownloader/files/23307950"

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
    Extracts the ZIP file content into a temporary folder, handles nested folders,
    and ensures the final structure is correct.
    """
    extracted_dir = os.path.join(target_dir, new_name)
    temp_dir = os.path.join(target_dir, "temp_extract")

    if os.path.exists(extracted_dir):
        print(f"The folder '{new_name}' already exists.")
        if not ask_user("Do you want to extract it again?"):
            return extracted_dir

    try:
        # Create the temporary folder
        os.makedirs(temp_dir, exist_ok=True)

        # Extract the ZIP file content into the temporary folder
        print(f"Extracting the file '{zip_path}' into '{temp_dir}'...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print(f"Extraction completed in '{temp_dir}'.")

        # Check the structure of the extracted content
        extracted_items = os.listdir(temp_dir)
        if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_dir, extracted_items[0])):
            # If there's only one folder inside the temporary directory
            inner_folder = os.path.join(temp_dir, extracted_items[0])
            if extracted_items[0] == new_name:
                # Move the inner folder directly to the target directory
                if os.path.exists(extracted_dir):
                    shutil.rmtree(extracted_dir)  # Remove the existing folder
                shutil.move(inner_folder, extracted_dir)
                print(f"Folder '{new_name}' moved to '{target_dir}'.")
            else:
                # Rename the inner folder to the desired name
                if os.path.exists(extracted_dir):
                    shutil.rmtree(extracted_dir)  # Remove the existing folder
                os.rename(inner_folder, extracted_dir)
                print(f"Folder renamed to '{new_name}' and moved to '{target_dir}'.")
        else:
            # If there are multiple files or folders, move everything to the target directory
            if os.path.exists(extracted_dir):
                shutil.rmtree(extracted_dir)  # Remove the existing folder
            os.rename(temp_dir, extracted_dir)
            print(f"Content moved to '{extracted_dir}'.")

        return extracted_dir
    except Exception as e:
        print(f"Error during extraction or renaming: {e}")
        exit(1)
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
    zip_file = os.path.join(DATABASE_DIR, "POLLEN73S.zip")
    renamed_dir_name = "POLLEN73S"

    # Step 1: Download the ZIP file
    download_database(DATABASE_URL, zip_file)

    # Step 2: Extract the ZIP file and rename the folder
    extracted_dir = extract_and_rename_zip(zip_file, DATABASE_DIR, renamed_dir_name)

    # Step 3: Clean up the compressed file (optional)
    clean_compressed_file(zip_file)

    print("Processing completed!")

if __name__ == "__main__":
    main()