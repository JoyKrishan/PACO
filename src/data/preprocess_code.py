import random
import os 
from pathlib import Path
from diff_processing import *
import json
import csv

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
# PATCHES_DIR = os.path.join(PROJECT_DIR, 'data', 'raw', 'Patches', 'Patches_others')
CSV_OUTPUT_PATH = os.path.join(PROJECT_DIR, 'data', 'csv_data')
OUTPUT_FILE_NAME = "patches.csv"

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

        except Exception as e:
            print(e)
            continue

        data.append({
            "id": unique_id,
            "file_name": file_name,
            # "bug_dataset": dataset_name,
            "buggy_code": buggy,
            "correct_patches": correct_patches,
            "incorrect_patches": incorrect_patches
            }),
        
    return data
    
def build_csv(data):
    csv_data  = []
    for item in data:
        # populate correct patches
        for patch in item["correct_patches"]:
            csv_data.append({
                "id": item["id"],
                "buggy_code": item["buggy_code"],
                "patch": patch,
                "patch_type": "correct"
            })
        # populate incorrect patches
        for patch in item["incorrect_patches"]:
            csv_data.append({
                "id": item["id"],
                "buggy_code": item["buggy_code"],
                "patch": patch,
                "patch_type": "incorrect"
            })
                
    # csv creation
    full_path = os.path.join(CSV_OUTPUT_PATH, OUTPUT_FILE_NAME)
    os.makedirs(CSV_OUTPUT_PATH, exist_ok=True)

    with open(full_path, 'w', newline='') as csvfile:
        fieldnames = ['id', 'file_name', 'buggy_code', 'patch', 'patch_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(csv_data)

    

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

    data = create_data_with_diff(correct_patch_dir, incorrect_patch_dir)
    build_csv(data)