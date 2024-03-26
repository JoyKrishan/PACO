from torchtext import data
from torchtext.data import BucketIterator
from transformers import AutoTokenizer

import os

from dotenv import load_dotenv, find_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv(find_dotenv())

def csv_train_test_split(save_path, df):
    """
        @params
        save_path: full path on where the train, val and test data will be saved
        df: full dataframe that should be divided and parsed to build the iterators

        @return
        train_iterator, valid_iterator, test_iterator, tokenizer
    """

    df = df.sample(frac=1).reset_index(drop=True)  

    df['numerical_label'] = df['label'].replace({'correct': 1, 'overfitting': -1})

    total_rows = len(df)
    first = int(0.8 * total_rows)
    second = int(0.9 * total_rows)

    train_df = df.iloc[:first]
    val_df = df.iloc[first:second]
    test_df = df.iloc[second:]

    train_path = os.path.join(save_path, "train.csv")
    val_path = os.path.join(save_path, "val.csv")
    test_path = os.path.join(save_path, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return (train_path, train_df), (val_path, val_df), (test_path, test_df)

def build_datasets(train_path, val_path, test_path, fields):
    train_dataset = data.TabularDataset(
        path= train_path,
        format="csv",
        skip_header=True,
        fields=fields
        )
    val_dataset = data.TabularDataset(
        path= val_path,
        format="csv",
        skip_header=True,
        fields=fields
        )
    test_dataset = data.TabularDataset(
        path= test_path,
        format="csv",
        skip_header=True,
        fields=fields
        )
    
    return train_dataset, val_dataset, test_dataset

def build_train_val_test_iter(save_path, df, device, tokenizer_name = "FacebookAI/roberta-base", max_len = 512, batch_size = 32):

    print(f"[INFO] Building train, val and test csv files at {save_path}")

    train_materials, val_materials, test_materials = csv_train_test_split(save_path, df)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    unk_index = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    max_seq_len = max_len

    # build the fields
    BUGGY_CODE = data.Field(use_vocab =False, tokenize = tokenizer.encode, pad_token=pad_index, unk_token=unk_index, fix_length=max_seq_len)
    PATCH_CODE = data.Field(use_vocab =False, tokenize= tokenizer.encode, pad_token=pad_index, unk_token=unk_index, fix_length=max_seq_len)
    LABEL = data.Field(use_vocab=False, sequential=False)

    fields = [("dataset", None),  # hard coded for convenience
        ("tool", None),
        ("buggy", BUGGY_CODE),
        ("patch", PATCH_CODE),
        ("label", None),
        ("numerical_label", LABEL)]
    
    print(f"[INFO] Building train, val and test tabular dataset")
    train_dataset, val_dataset, test_dataset = build_datasets(train_materials[0], val_materials[0], test_materials[0], fields)

    print(f"** Number of training examples: {len(train_dataset.examples)}")
    print(f"** Number of validation examples: {len(val_dataset.examples)}")
    print(f"** Number of testing examples: {len(test_dataset.examples)}")  

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, val_dataset, test_dataset), 
    batch_size = batch_size, 
    sort_key = lambda x : len(x.__dict__),
    sort_within_batch = False,
    device = device)

    print(f"Batch size: {batch_size}")
    return train_iterator, valid_iterator, test_iterator, tokenizer