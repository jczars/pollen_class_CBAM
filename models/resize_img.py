import os
import cv2
from tqdm import tqdm
import argparse

def resize_images_in_place(src, target_size=(224, 224), file_type='png', verbose=0):
    """
    Verifies if images in the source directory have different sizes and resizes them to the target size (224x224).
    The resized images are saved in the same directory, overwriting the original files.

    Parameters:
    - src (str): Source directory containing subdirectories with images.
    - target_size (tuple): Target dimensions for resizing (width, height). Default is (224, 224).
    - file_type (str): Type of image files to process (e.g., png, jpg). Default is 'png'.
    - verbose (int): Verbosity level (0 for silent, 1 for detailed messages). Default is 0.
    """
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
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Resize images in place to a target size.")
    parser.add_argument("src", type=str, help="Source directory containing subdirectories with images.")
    parser.add_argument("--target_size", type=int, nargs=2, default=[224, 224],
                        help="Target size for resizing (width height). Default is 224 224.")
    parser.add_argument("--file_type", type=str, default="png",
                        help="Type of image files to process (e.g., png, jpg). Default is png.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output. Default is False.")

    # Parse arguments
    args = parser.parse_args()

    # Convert target_size to a tuple
    target_size = tuple(args.target_size)

    # Call the resize function with parsed arguments
    resize_images_in_place(
        src=args.src,
        target_size=target_size,
        file_type=args.file_type,
        verbose=1 if args.verbose else 0
    )


if __name__ == "__main__":
    main()