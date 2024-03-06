import random
import os 
from pathlib import Path
from diff_processing import *
import json

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
# PATCHES_DIR = os.path.join(PROJECT_DIR, 'data', 'raw', 'Patches', 'Patches_others')

json_data = []
used_numbers = set()
def generate_unique_random_number():
    """Generates a unique 4-digit number while ensuring no repeats."""
    while True:
        number = random.randint(1000, 9999)  # Range for 4-digit numbers
        if str(number) not in used_numbers:
            used_numbers.add(str(number))
            return number


def find_all_files_with_bug_id(directory_path, bug_id):
    """Finds files containing bug_id in a given directory."""

    patch_files_for_specific_bug = []
    for filename in os.listdir(directory_path):
        if re.search(r"\b" + bug_id + r"\b", filename):
            patch_files_for_specific_bug.append(filename)

    return patch_files_for_specific_bug

def get_all_patches(patches, patch_dir):
    all_patches = []
    for patch_file_name in patches:
        toolname = patch_file_name.split('-')[3] # excluding for now
        correct = get_diff_files_frag(os.path.join(patch_dir, patch_file_name),type='patched')
        all_patches.append(correct)
    
    return all_patches

def create_data_with_diff(correct_patch_dir, incorrect_patch_dir):
 
    write_file = os.path.join(PROCESSED_DATA_DIR, 'paco_dataset_diff_files_frag.json')
    all_correct_patch_files = os.listdir(correct_patch_dir)
    all_incorrect_patch_files = os.listdir(incorrect_patch_dir)
    all_bug_ids = set()
    data = []
    for file_name in all_correct_patch_files:
        unique_id = generate_unique_random_number()
        names = file_name.split("-")
        bug_id = names[1] + "-" + names[2]
        if bug_id in all_bug_ids:
            continue
        all_bug_ids.add(bug_id)
        try:
            buggy = get_diff_files_frag(os.path.join(correct_patch_dir, file_name),type='buggy') # find the buggy code from one item

            correct_patch_files_for_specific_bug = find_all_files_with_bug_id(correct_patch_dir, bug_id) # list of all the correct patches of one specific bug

            correct_patches = get_all_patches(correct_patch_files_for_specific_bug, correct_patch_dir)

            incorrect_patch_files_for_specific_bug = find_all_files_with_bug_id(incorrect_patch_dir, bug_id) 

            incorrect_patches = get_all_patches(incorrect_patch_files_for_specific_bug, incorrect_patch_dir)

        except Exception:
            print(Exception)
            continue

        data.append({
            "id": unique_id,
            "file_name": file_name,
            # "bug_dataset": dataset_name,
            "buggy_code": buggy,
            "correct_patches": correct_patches,
            "incorrect_patches": incorrect_patches
            }),
        print(f"Patches appended for {bug_id}!")
    with open(write_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)

    print(all_bug_ids)
    print(f"Total bugs found: {len(all_bug_ids)}")
    print("Data written to bug_data.json successfully!")
    

"""Commenting this functionality for now
def create_data_wholefile(path_patched):
    with open('../data/pre_data_whole.txt','w+') as f:
        data = ''
        for root,dirs,files in os.walk(path_patched):
            for file in files:
                if file == 'QUICKSORT.java':
                    buggy = get_whole('/Users/haoye.tian/Documents/University/data/quixbugs_wholefile/quicksort/QUICKSORT_BUG.java')
                    patched = get_whole(os.path.join(root, file))
                    bug_id = root.split('quicksort/')[1].split('/java_programs')[0]
                    label_temp = '1'
                    sample = label_temp + '<ml>' + bug_id + '<ml>' + buggy + '<ml>' + patched
                    data += sample + '\n'
        f.write(data)"""


if __name__ == '__main__':
    correct_patch_dir = os.path.join(PROJECT_DIR, 'data', 'raw', 'custom_patches', 'correct')
    incorrect_patch_dir = os.path.join(PROJECT_DIR, 'data', 'raw', 'custom_patches', 'overfitting')

    create_data_with_diff(correct_patch_dir, incorrect_patch_dir)