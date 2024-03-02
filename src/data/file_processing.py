import os, shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
PATCHES_DIR = os.path.join(PROJECT_DIR, 'data', 'raw', 'Patches', 'Patches_ICSE')

def copy_to_desired_dir(directory): 
    """
    Copies .patches to the desired directory
    Args:
        directory (path, string): specified directory

    Returns:
        None
    """
    destination_dir = directory
    count = 0
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".patch"):
                source_path = os.path.join(root, filename)
                destination_path = os.path.join(destination_dir, filename)
                shutil.move(source_path, destination_path) 
                count += 1
    print(f"Moved {count} patches into {os.path.basename(destination_dir)}")


def clean_dir(directtory_to_clean):
    """
    Cleans the .patches files from the specified directory. (Used only if the copying went wrong)
    Args:
        directory_to_clean (path, string): specified directory

    Returns:
        None
    """
    count = 0
    for file in os.listdir(directtory_to_clean):
        full_path = os.path.join(directtory_to_clean, file)
        if file.endswith('.patch') and os.path.isfile(full_path):
            os.remove(full_path)
            count += 1
    print(f"Removed {count} patches from {os.path.basename(directtory_to_clean)}")

if __name__ == "__main__":
    incorrect_patches_dir = os.path.join(PATCHES_DIR, 'Doverfitting')

    copy_to_desired_dir(incorrect_patches_dir)
    # copy_to_desired_dir(correct_same_patches_dir)