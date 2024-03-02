import json
from pathlib import Path
import os

def extract_and_write_code(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    with open(output_file, 'w') as f:

        for example in data:
            f.write(' '.join(example['buggy_code'].split(' ')[1:]))

            for patch in example['correct_patches']:
                f.write(' '.join(patch.split(' ')[1:]))  # Write correct patches

            for patch in example['incorrect_patches']:
                f.write(' '.join(patch.split(' ')[1:]))  # Write incorrect patches   
    
if __name__ == "__main__":

    PROJECT_DIR = Path(__file__).resolve().parents[2]
    PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
    json_file = os.path.join(PROCESSED_DATA_DIR, 'paco_dataset_diff_files.json')
    output_file = os.path.join(PROCESSED_DATA_DIR, 'code_corpus.txt')
    code_snippets = extract_and_write_code(json_file)
