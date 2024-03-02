import random
import os 
from pathlib import Path
from diff_processing import *
import json

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
PATCHES_DIR = os.path.join(PROJECT_DIR, 'data', 'raw', 'Patches', 'Patches_others')

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
        if bug_id in filename:
            patch_files_for_specific_bug.append(filename)

    return patch_files_for_specific_bug

def get_all_patches(patches, patch_dir):
    all_patches = []
    for patch_file_name in patches:
        toolname = patch_file_name.split('-')[3] # excluding for now
        correct = get_diff_files(os.path.join(patch_dir, patch_file_name),type='patched')
        all_patches.append(correct)
    
    return all_patches

def create_data_with_diff(correct_patch_dir, incorrect_patch_dir):
 
    write_file = os.path.join(PROCESSED_DATA_DIR, 'paco_dataset_diff_files.json')
    all_correct_patch_files = os.listdir(correct_patch_dir)
    all_incorrect_patch_files = os.listdir(incorrect_patch_dir)
    data = []
    for file_name in all_correct_patch_files:
        unique_id = generate_unique_random_number()
        names = file_name.split("-")
        bug_id = names[1] + "-" + names[2]

        try:
            buggy = get_diff_files(os.path.join(correct_patch_dir, file_name),type='buggy') # find the buggy code from one item

            correct_patch_files_for_specific_bug = find_all_files_with_bug_id(correct_patch_dir, bug_id) # list of all the correct patches of one specific bug

            correct_patches = get_all_patches(correct_patch_files_for_specific_bug, correct_patch_dir)

            incorrect_patch_files_for_specific_bug = find_all_files_with_bug_id(incorrect_patch_dir, bug_id) 

            incorrect_patches = get_all_patches(incorrect_patch_files_for_specific_bug, incorrect_patch_dir)

        except Exception:
            print(Exception)
            continue

        all_correct_patch_files =  [dupli_file for dupli_file in all_correct_patch_files if dupli_file not in correct_patch_files_for_specific_bug]

        data.append({
            "id": unique_id,
            # "bug_dataset": dataset_name,
            "buggy_code": buggy,
            "correct_patches": correct_patches,
            "incorrect_patches": incorrect_patches
            }),
        print(f"Patches appended for {bug_id}!")
    with open(write_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)

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


def create_train_data5(path_patch_kui):
    with open('../data/train_data5.txt','w+') as f:
        data = ''
        for root,dirs,files in os.walk(path_patch_kui):
            if files == ['.DS_Store']:
                continue
            # files = sorted(files,key=lambda x:int(x.split('-')[1].split('.')[0]))
            for file in files:
                # if root.startswith('../data/kui_Patches/Patches_train/Bears'):
                #     pass
                if file.endswith('txt') or file.endswith('patch'):
                    if file.endswith('.patch'):
                        bug_id = '_'.join([root.split('/')[-2],root.split('/')[-1], file])
                    else:
                        bug_id = '_'.join([root.split('/')[-1],file])
                    try:
                        # if bug_id.endswith('Bears-114.txt'):
                            # pass
                        buggy = get_diff_files(os.path.join(root,file),type='buggy')
                        patched = get_diff_files(os.path.join(root, file), type='patched')
                    except Exception as e:
                        print(e)
                        continue
                    label_temp = '1'
                    sample = label_temp + '<ml>' + bug_id + '<ml>' + buggy + '<ml>' + patched
                    data += sample + '\n'
                    # with open(root+'/pre_data_kui.txt', 'a+') as f:
                    #     f.write(sample)
        f.write(data)

def create_train_data5_frag(path_patch_train):
    with open('../data/experiment1/train_data5_frag_error.txt','w+') as f:
        data = ''
        for root,dirs,files in os.walk(path_patch_train):
            if files == ['.DS_Store']:
                continue
            # files = sorted(files,key=lambda x:int(x.split('-')[1].split('.')[0]))
            for file in files:
                # if root.startswith('../data/kui_Patches/Patches_train/Bears'):
                #     pass
                if file.endswith('txt') or file.endswith('patch'):
                    if file.endswith('.patch'):
                        bug_id = '_'.join([root.split('/')[-2],root.split('/')[-1], file])
                    else:
                        bug_id = '_'.join([root.split('/')[-1],file])
                    try:
                        # if bug_id.endswith('Bears-114.txt'):
                            # pass
                        buggy = get_diff_files_frag(os.path.join(root,file),type='buggy')
                        patched = get_diff_files_frag(os.path.join(root, file), type='patched')
                    except Exception as e:
                        print(e)
                        continue
                    if buggy == '' or patched == '':
                        print('null patch')
                        continue
                    label_temp = '1'
                    sample = label_temp + '<ml>' + bug_id + '<ml>' + buggy + '<ml>' + patched
                    data += sample + '\n'
                    # with open(root+'/pre_data_kui.txt', 'a+') as f:
                    #     f.write(sample)
        f.write(data)




def create_train_data_for_incorrect(path_incorrect):
    with open('../data/experiment1/train_data5_frag_incorrect.txt','w+') as f:
        data = ''

        # patch from kui
        for root, dirs, files in os.walk(path_incorrect):
            if files == ['.DS_Store']:
                continue
            # files = sorted(files,key=lambda x:int(x.split('-')[1].split('.')[0]))
            if files == []:
                continue
            # get label
            label_temp = root.split('/')[-1]
            label = '0' if ('P' in label_temp) else '1'
            if label == '1':
                continue
            bugy_all = ''
            patched_all = ''
            for file in files:
                if file.endswith('txt'):
                    bug_id = '_'.join([root.split('/')[-1], file])
                    try:
                        buggy = get_diff_files_frag(os.path.join(root, file), type='buggy')
                        patched = get_diff_files_frag(os.path.join(root, file), type='patched')
                    except Exception as e:
                        print(e)
                        continue
                    bugy_all += buggy
                    patched_all += patched
            if bugy_all == '' or patched_all == '':
                continue
            sample = label + '<ml>' + bug_id + '<ml>' + bugy_all + '<ml>' + patched_all
            data += sample + '\n'

        f.write(data)



if __name__ == '__main__':
    correct_patch_dir = os.path.join(PROJECT_DIR, 'data', 'raw', 'custom_patches', 'correct')
    incorrect_patch_dir = os.path.join(PROJECT_DIR, 'data', 'raw', 'custom_patches', 'overfitting')

    create_data_with_diff(correct_patch_dir, incorrect_patch_dir)
    # create_train_data5(path_patch_train)

    # correct patches
    # create_train_data5_frag(path_patch_train)
    # create_train_data5_for_cc2v(path_patch_train)

    # incorrect patches
    # create_train_data_for_incorrect(path_incorrect)