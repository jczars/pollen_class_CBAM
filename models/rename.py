import os
import re

def rename_subdirectories(base_dir):
    """
    Renames subdirectories in the specified directory by removing numbers and converting to lowercase.
    """
    print(f"DEBUG: Base directory: {base_dir}")
    
    # Step 1: Check if the base directory exists
    if not os.path.exists(base_dir):
        print(f"ERROR: The directory '{base_dir}' does not exist.")
        return

    # Step 2: List all items in the base directory
    print("DEBUG: Listing subdirectories...")
    renamed_count = 0
    for item in os.listdir(base_dir):
        old_path = os.path.join(base_dir, item)
        
        if os.path.isdir(old_path):  # Check if it's a directory
            print(f"DEBUG: Found subdirectory: {item}")
            
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
            else:
                print(f"DEBUG: Subdirectory '{item}' does not match the renaming pattern.")
        else:
            print(f"DEBUG: Item '{item}' is not a directory (skipping).")

    if renamed_count == 0:
        print("DEBUG: No subdirectories were found to rename. Ensure the directory structure and naming patterns are correct.")
    else:
        print(f"DEBUG: Total subdirectories renamed: {renamed_count}")

# Example usage
if __name__ == "__main__":
    # Define the base directory where the extracted files are located
    output_dir = "./BD/CPD1_Cr"  # Adjust this path based on your setup

    # Call the rename_subdirectories function
    rename_subdirectories(output_dir)