import code_tokenize as ctok
import json
from pathlib import Path
import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')

def ai_tokenize_code(code_corpus_path):

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["[CLS]", "[SEP]", "[PAD]"])
    tokenizer.train(files= [code_corpus_path], trainer = trainer)

    return tokenizer

"""Not using this function for now: It is buggy
def tokenize_code(code, lang, syn_err = "ignore"):
    tokenized_code = None 
    try:
        tokenized_code = ctok.tokenize(code, lang = lang, syntax_error = syn_err)
    except Exception as err:
        print(err)

    return tokenized_code
    """

def tokenize_code_json(json_file, output_file, tokenizer):
    with open(json_file, 'r') as f:
        data = json.load(f)

    for sample in data:
        sample["buggy_code_tokens"] = tokenizer.encode(" ".join(sample["buggy_code"].split(' ')[1:])).tokens
        sample["correct_patches_tokens"] = []
        sample["incorrect_patches_tokens"] = []
        for i, patch in enumerate(sample["correct_patches"]):
            sample["correct_patches_tokens"].append(tokenizer.encode(" ".join(patch.split(' ')[1:])).tokens)

        for i, patch in enumerate(sample["incorrect_patches"]):
            sample["incorrect_patches_tokens"].append(tokenizer.encode(" ".join(patch.split(' ')[1:])).tokens)
        
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":

    code_corpus = os.path.join(PROCESSED_DATA_DIR, 'code_corpus.txt')
    patch_json_file = os.path.join(PROCESSED_DATA_DIR, 'paco_dataset_diff_files_frag.json')
    out_file = os.path.join(PROCESSED_DATA_DIR, 'tokenized_dataset.json')

    ctokenizer = ai_tokenize_code(code_corpus)
    tokenize_code_json(patch_json_file, out_file, ctokenizer)