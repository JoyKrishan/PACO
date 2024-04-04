from dotenv import load_dotenv, find_dotenv
import sys, os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

load_dotenv(find_dotenv())
project_root = os.getenv('PROJECT_ROOT')
sys.path.insert(0, project_root) 

import click
import torch
import torch.optim as optim
import optuna
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
from prepare_data_for_models import build_train_val_test_iter
from supconloss_with_cosine import SupConLossWithConsine
from src.models.lstm_code_encoder import LSTMCodeEncoder
from torch.utils.tensorboard import SummaryWriter
import random, time


def create_embedding_plot_with_best_model(model_path, iterator, tokenizer, writer):
    best_code_encoder = LSTMCodeEncoder(tokenizer.vocab_size, 512, 512, num_layers=1).to(device)
    best_code_encoder = nn.DataParallel(best_code_encoder)
    best_code_encoder.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        data_list = list(iterator)  # Potential memory concerns for large datasets
        random_index = random.randint(0, len(data_list) - 1)
        random_batch = data_list[random_index] 

        bug_embd = best_code_encoder(random_batch.buggy.T)
        patch_embd = best_code_encoder(random_batch.patch.T)
        labels = random_batch.numerical_label

        # Separate embeddings based on labels
        correct_mask = labels == 1 
        incorrect_mask = labels == -1 

        bug_embd_correct = bug_embd[correct_mask] 
        bug_embd_incorrect = bug_embd[incorrect_mask]
        patch_embd_incorrect = patch_embd[incorrect_mask]

        correct_embeddings = torch.cat([bug_embd_correct, patch_embd_correct], dim=0)
        incorrect_embeddings = torch.cat([bug_embd_incorrect, patch_embd_incorrect], dim=0)

        correct_metadata = ['buggy'] * bug_embd_correct.shape[0] + ['correct_patch'] * patch_embd_correct.shape[0]
        incorrect_metadata = ['buggy'] * bug_embd_incorrect.shape[0] + ['incorrect_patch'] * patch_embd_incorrect.shape[0]

        # Add embeddings to TensorBoard
        writer.add_embedding(correct_embeddings.reshape(correct_embeddings.shape[0], -1), metadata=correct_metadata, tag='correct', global_step=0)
        writer.add_embedding(incorrect_embeddings.reshape(incorrect_embeddings.shape[0], -1), metadata=incorrect_metadata, tag='incorrect', global_step=0)

    writer.close()

@click.command()
@click.option('--dataset', type=click.Choice(['small', 'large', 'all']), 
              prompt='Dataset to test', required=True)
@click.option('--set', type=click.Choice(['train', 'val', 'test']), 
              prompt='Which set to test', required=True)
def main(dataset, set):

    if dataset == 'all':
        train_itr, val_itr, test_itr, tokenizer, batch_size = build_train_val_test_iter(all_dataset_output, all_patches_df, device)
        writer_path =  os.path.join(notebook_path, "runs/all_dataset/all_runs/")
        hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_all")
        if set == 'train':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_train")
            create_embedding_plot_with_best_model(os.path.join(all_dataset_best_model_save_path, "best_model.pth"), train_itr, tokenizer, hyper_parameter_writer)
        if set == 'val':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_val")
            create_embedding_plot_with_best_model(os.path.join(all_dataset_best_model_save_path, "best_model.pth"), val_itr, tokenizer, hyper_parameter_writer)
        if set == 'test':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_test")
            create_embedding_plot_with_best_model(os.path.join(all_dataset_best_model_save_path, "best_model.pth"), test_itr, tokenizer, hyper_parameter_writer)

    if dataset == 'small':
        train_itr, val_itr, test_itr, tokenizer, batch_size = build_train_val_test_iter(small_dataset_output, small_patches_df, device)
        writer_path =  os.path.join(notebook_path, "runs/small_dataset/all_runs/")
        if set == 'train':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_train")
            create_embedding_plot_with_best_model(os.path.join(small_dataset_best_model_save_path, "best_model.pth"), train_itr, tokenizer, hyper_parameter_writer)
        if set == 'val':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_val")
            create_embedding_plot_with_best_model(os.path.join(small_dataset_best_model_save_path, "best_model.pth"), val_itr, tokenizer, hyper_parameter_writer)
        if set == 'test':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_test")
            create_embedding_plot_with_best_model(os.path.join(small_dataset_best_model_save_path, "best_model.pth"), test_itr, tokenizer, hyper_parameter_writer)

    if dataset == 'large':
        train_itr, val_itr, test_itr, tokenizer, batch_size = build_train_val_test_iter(large_dataset_output, large_patches_df, device)
        writer_path =  os.path.join(notebook_path, "runs/large_dataset/all_runs/")
        if set == 'train':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_train")
            create_embedding_plot_with_best_model(os.path.join(large_dataset_best_model_save_path, "best_model.pth"), train_itr, tokenizer, hyper_parameter_writer)
        if set == 'val':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_val")
            create_embedding_plot_with_best_model(os.path.join(large_dataset_best_model_save_path, "best_model.pth"), val_itr, tokenizer, hyper_parameter_writer)
        if set == 'test':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_test")
            create_embedding_plot_with_best_model(os.path.join(large_dataset_best_model_save_path, "best_model.pth"), test_itr, tokenizer, hyper_parameter_writer)


if __name__ == "__main__":
    seed = 42  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    notebook_path = os.path.join(project_root, "notebooks")
    data_dir = os.environ.get("DATA_DIR")
    dataset_fullpath = os.path.join(project_root, data_dir, "output")
    # load dataset output to build the train, val and test CSVs
    all_dataset_output = os.path.join(dataset_fullpath, "model_output", "all_patches")
    large_dataset_output = os.path.join(dataset_fullpath, "model_output", "large_patches")
    small_dataset_output = os.path.join(dataset_fullpath, "model_output", "small_patches")

    notebook_path = os.path.join(project_root, "notebooks")

    # load the dataframes
    all_patches_df = pd.read_csv(os.path.join(dataset_fullpath, "all-patches.csv"))
    large_patches_df = pd.read_csv(os.path.join(dataset_fullpath, "large-patches.csv"))
    small_patches_df = pd.read_csv(os.path.join(dataset_fullpath, "small-patches.csv"))

    # best snapshots
    all_dataset_best_model_save_path = os.path.join(project_root, "notebooks", "runs", "all_dataset", "model")
    large_dataset_best_model_save_path = os.path.join(project_root, "notebooks", "runs", "large_dataset", "model")
    small_dataset_best_model_save_path = os.path.join(project_root, "notebooks", "runs", "small_dataset", "model")

    main()