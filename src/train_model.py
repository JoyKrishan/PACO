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


def train(model, train_iterator, valid_iterator, batch_size, trial, n_epochs, writer_path):
    print(f'Trial Number: {trial.number}\nStart time: {time.strftime("%H:%M:%S")}')
    nb_epochs = n_epochs

    print('------------------- MODEL TRAINING STARTED -------------------')
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam'])

    train_steps = len(train_iterator.dataset) // batch_size
    val_steps = len(valid_iterator.dataset) // batch_size

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate) 
    criterion = SupConLossWithConsine(device=device)
    
    writer = SummaryWriter(writer_path + f"{trial.number}")
    model.train()
    for epoch in range(nb_epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0

        # train_correct = 0  not using it now since there is no patch classifier
        # val_correct = 0

        for batch in train_iterator:
            buggy_tensor, patch_tensor, labels = batch.buggy.T, batch.patch.T, batch.numerical_label # data already in device

            optimizer.zero_grad()
            buggy_embd = model(buggy_tensor)
            patch_embd = model(patch_tensor)
            loss = criterion(buggy_embd, patch_embd, labels) # buggy_embd, patch_embd, label
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss

        with torch.no_grad():
            for batch in valid_iterator:
                buggy_tensor, patch_tensor, labels = batch.buggy.T, batch.patch.T, batch.numerical_label
                buggy_embd = model(buggy_tensor)
                patch_embd = model(patch_tensor)

                loss = criterion(buggy_embd, patch_embd, labels) # buggy_embd, patch_embd, label
                
                epoch_val_loss += loss

        mean_train_loss = epoch_train_loss / train_steps
        mean_val_loss = epoch_val_loss / val_steps

        writer.add_scalar("training_loss", mean_train_loss, epoch + 1)
        writer.add_scalar("validation_loss", mean_val_loss, epoch + 1)

    writer.add_hparams({'lr': learning_rate}, {'train_loss': mean_train_loss, 'val_loss': mean_val_loss})
    print(f'Trial{trial.number}\nEnd time: {time.strftime("%H:%M:%S")}')
    writer.close()
    return mean_val_loss


def create_and_run_study(model_savepath, train_itr, val_itr, n_trials, tokenizer, batch_size, epochs, writer_path):
    study = optuna.create_study(direction='minimize')   # Aim to minimize  validation loss

    def save_model(model, filename):
        torch.save(model.state_dict(), os.path.join(model_savepath, filename))
        
    def objective(trial):
        code_encoder = LSTMCodeEncoder(tokenizer.vocab_size, 512, 512, num_layers=1)
        code_encoder.to(device)
        if torch.cuda.device_count() > 1:
            code_encoder = nn.DataParallel(code_encoder)
   
        val_loss = train(code_encoder, train_itr, val_itr, batch_size, trial, epochs, writer_path) # bad practice will change later

        if trial.number == 0 or val_loss < study.best_value:
            save_model(code_encoder, "best_model.pth")
        return val_loss  
    
    study.optimize(objective, n_trials=n_trials)


@click.command()

@click.option('--dataset', type=click.Choice(['small', 'large', 'all']), 
              prompt='Dataset to train', required=True)
@click.option('--epochs', type=int, 
              prompt='Number of epochs to train', required=True)
@click.option('--trials', type=int, 
              prompt='Number of trials', required=True)
def main(dataset, epochs, trials):
    if dataset == 'all':
        print("PROCESSING ON THE ALL DATASET")
        train_itr, val_itr, test_itr, tokenizer, batch_size = build_train_val_test_iter(all_dataset_output, all_patches_df, device)
        writer_path = os.path.join(notebook_path, "runs/all_dataset/all_runs/")
        create_and_run_study(all_dataset_best_model_save_path, train_itr, val_itr, trials, tokenizer, batch_size, epochs, writer_path)

    if dataset == 'large':
        print("PROCESSING ON THE LARGE DATASET")
        train_itr, val_itr, test_itr, tokenizer, batch_size = build_train_val_test_iter(large_dataset_output, large_patches_df, device)
        writer_path =  os.path.join(notebook_path, "runs/large_dataset/all_runs/")
        create_and_run_study(large_dataset_best_model_save_path, train_itr, val_itr, trials, tokenizer, batch_size, epochs, writer_path)

    if dataset == 'small':
        print("PROCESSING ON THE SMALL DATASET")
        train_itr, val_itr, test_itr, tokenizer, batch_size = build_train_val_test_iter(small_dataset_output, small_patches_df, device)
        writer_path = os.path.join(notebook_path, "runs/small_dataset/all_runs/")
        create_and_run_study(small_dataset_best_model_save_path, train_itr, val_itr, trials, tokenizer, batch_size, epochs, writer_path)

if __name__ == "__main__":
    seed = 42  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = os.environ.get("DATA_DIR")
    dataset_fullpath = os.path.join(project_root, data_dir, "output")

    # load dataset output to build the train, val and test CSVs
    all_dataset_output = os.path.join(dataset_fullpath, "model_output", "all_patches")
    large_dataset_output = os.path.join(dataset_fullpath, "model_output", "large_patches")
    small_dataset_output = os.path.join(dataset_fullpath, "model_output", "large_patches")

    notebook_path = os.path.join(project_root, "notebooks")

    # best snapshots
    all_dataset_best_model_save_path = os.path.join(project_root, "notebooks", "runs", "all_dataset", "model")
    large_dataset_best_model_save_path = os.path.join(project_root, "notebooks", "runs", "large_dataset", "model")
    small_dataset_best_model_save_path = os.path.join(project_root, "notebooks", "runs", "small_dataset", "model")

    # load the dataframes
    all_patches_df = pd.read_csv(os.path.join(dataset_fullpath, "all-patches.csv"))
    large_patches_df = pd.read_csv(os.path.join(dataset_fullpath, "large-patches.csv"))
    small_patches_df = pd.read_csv(os.path.join(dataset_fullpath, "small-patches.csv"))

    main()