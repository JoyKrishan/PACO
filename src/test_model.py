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
import random
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score 


def create_train_test_data(code_encoder, train_itr, val_itr, test_itr):

    cosine_sim = nn.CosineSimilarity(dim = 2)

    def create_x_y_on_iterator(iterator):
        X = []
        Y = []

        for batch in iterator:
            buggy_embeddings = code_encoder(batch.buggy.T)
            patch_embeddings = code_encoder(batch.patch.T)

            sim_score = cosine_sim(buggy_embeddings, patch_embeddings)
            X.append(sim_score.cpu().detach().numpy())           
            Y.append(batch.numerical_label.cpu().detach().numpy())
        
        X = np.vstack(X)
        Y = np.concatenate(Y)
        return X, Y

    X_train, Y_train = create_x_y_on_iterator(train_itr)
    X_val, Y_val = create_x_y_on_iterator(val_itr)
    X_test, Y_test = create_x_y_on_iterator(test_itr)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def patch_classifier(model_class, model_name, X_train, Y_train, X_val, Y_val, X_test, Y_test):

    model = model_class()

    model.fit(X_train, Y_train)
    predictions_train = model.predict(X_test) 
    predictions_train_proba = model.predict_proba(X_test)[:, 1]

    # Calculate Metrics
    accuracy = accuracy_score(Y_test, predictions_train)
    precision = precision_score(Y_test, predictions_train)
    recall = recall_score(Y_test, predictions_train)
    f1 = f1_score(Y_test, predictions_train)
    auc = roc_auc_score(Y_test, predictions_train_proba) 

    print(f"***************Patch CLassifier as {model_name}*********************")
    print("Performance On the TEST SET...........")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"AUC: {auc:.3f}")


def create_embedding_plot_with_best_model(model_path, iterator, tokenizer, writer, embedding):

    best_code_encoder = LSTMCodeEncoder(tokenizer.vocab_size, 512, 512, num_layers=1).to(device)
    best_code_encoder = nn.DataParallel(best_code_encoder)
    best_code_encoder.load_state_dict(torch.load(model_path))

    if embedding == True:
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
            patch_embd_correct = patch_embd[correct_mask]
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
    return best_code_encoder


@click.command()
@click.option('--dataset', type=click.Choice(['small', 'large', 'all']), 
              prompt='Dataset to test', required=True)
@click.option('--set', type=click.Choice(['train', 'val', 'test']), 
              prompt='Which set to run on (create embedding or print performance)', required=True)
@click.option('--metrics', type=click.Choice(['yes', 'no']), required=False, prompt='RUN METRICS TEST: Select only if Test set selected, if no selected embedding will be created')
def main(dataset, set, metrics):

    if dataset == 'all':

        train_itr, val_itr, test_itr, tokenizer, batch_size = build_train_val_test_iter(all_dataset_output, all_patches_df, device)
        writer_path =  os.path.join(notebook_path, "runs/all_dataset/all_runs/")

        if set == 'train':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_train")
            best_code_encoder = create_embedding_plot_with_best_model(os.path.join(all_dataset_best_model_save_path, 
                                                                                    "best_model.pth"), train_itr, tokenizer, hyper_parameter_writer, True)     
        if set == 'val':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_val")
            best_code_encoder = create_embedding_plot_with_best_model(os.path.join(all_dataset_best_model_save_path, 
                                                                                   "best_model.pth"), val_itr, tokenizer, hyper_parameter_writer, True)
        if set == 'test' and metrics == 'no':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_test")
            best_code_encoder = create_embedding_plot_with_best_model(os.path.join(all_dataset_best_model_save_path, 
                                                                                   "best_model.pth"), test_itr, tokenizer, hyper_parameter_writer, True)
        if set == 'test' and metrics == 'yes':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_test")
            best_code_encoder = create_embedding_plot_with_best_model(os.path.join(all_dataset_best_model_save_path, 
                                                                                   "best_model.pth"), test_itr, tokenizer, hyper_parameter_writer, False)

            X_train, Y_train, X_val, Y_val, X_test, Y_test = create_train_test_data(best_code_encoder, train_itr, val_itr, test_itr)
            patch_classifier(LogisticRegression, "LogisticRegression", X_train, Y_train, X_val, Y_val, X_test, Y_test)
            patch_classifier(DecisionTreeClassifier, "DecisionTreeClassifier", X_train, Y_train, X_val, Y_val, X_test, Y_test)
            patch_classifier(RandomForestClassifier, "RandomForestClassifier", X_train, Y_train, X_val, Y_val, X_test, Y_test)

    if dataset == 'small':
        train_itr, val_itr, test_itr, tokenizer, batch_size = build_train_val_test_iter(small_dataset_output, small_patches_df, device)
        writer_path =  os.path.join(notebook_path, "runs/small_dataset/all_runs/")
        if set == 'train':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_train")
            best_code_encoder = create_embedding_plot_with_best_model(os.path.join(small_dataset_best_model_save_path, 
                                                               "best_model.pth"), train_itr, tokenizer, hyper_parameter_writer, True)
        if set == 'val':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_val")
            best_code_encoder = create_embedding_plot_with_best_model(os.path.join(small_dataset_best_model_save_path, 
                                                               "best_model.pth"), val_itr, tokenizer, hyper_parameter_writer, True)
        if set == 'test' and metrics == 'no':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_test")
            best_code_encoder = create_embedding_plot_with_best_model(os.path.join(small_dataset_best_model_save_path, 
                                                               "best_model.pth"), test_itr, tokenizer, hyper_parameter_writer, True)
        if set == 'test' and metrics == 'yes':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_test")
            best_code_encoder = create_embedding_plot_with_best_model(os.path.join(small_dataset_best_model_save_path, 
                                                                                   "best_model.pth"), test_itr, tokenizer, hyper_parameter_writer, False)

            X_train, Y_train, X_val, Y_val, X_test, Y_test = create_train_test_data(best_code_encoder, train_itr, val_itr, test_itr)
            patch_classifier(LogisticRegression, "LogisticRegression", X_train, Y_train, X_val, Y_val, X_test, Y_test)
            patch_classifier(DecisionTreeClassifier, "DecisionTreeClassifier", X_train, Y_train, X_val, Y_val, X_test, Y_test)
            patch_classifier(RandomForestClassifier, "RandomForestClassifier", X_train, Y_train, X_val, Y_val, X_test, Y_test)


    if dataset == 'large':
        train_itr, val_itr, test_itr, tokenizer, batch_size = build_train_val_test_iter(large_dataset_output, large_patches_df, device)
        writer_path =  os.path.join(notebook_path, "runs/large_dataset/all_runs/")
        if set == 'train':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_train")
            best_code_encoder = create_embedding_plot_with_best_model(os.path.join(large_dataset_best_model_save_path, 
                                                               "best_model.pth"), train_itr, tokenizer, hyper_parameter_writer, True)
        if set == 'val':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_val")
            best_code_encoder = create_embedding_plot_with_best_model(os.path.join(large_dataset_best_model_save_path, 
                                                               "best_model.pth"), val_itr, tokenizer, hyper_parameter_writer, True)
        if set == 'test' and metrics == 'no':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_test")
            best_code_encoder = create_embedding_plot_with_best_model(os.path.join(large_dataset_best_model_save_path, 
                                                               "best_model.pth"), test_itr, tokenizer, hyper_parameter_writer, True)
            
        if set == 'test' and metrics == 'yes':
            hyper_parameter_writer = SummaryWriter(writer_path + "hyperparameter_test")
            best_code_encoder = create_embedding_plot_with_best_model(os.path.join(large_dataset_best_model_save_path, 
                                                                                   "best_model.pth"), test_itr, tokenizer, hyper_parameter_writer, False)

            X_train, Y_train, X_val, Y_val, X_test, Y_test = create_train_test_data(best_code_encoder, train_itr, val_itr, test_itr)
            patch_classifier(LogisticRegression, "LogisticRegression", X_train, Y_train, X_val, Y_val, X_test, Y_test)
            patch_classifier(DecisionTreeClassifier, "DecisionTreeClassifier", X_train, Y_train, X_val, Y_val, X_test, Y_test)
            patch_classifier(RandomForestClassifier, "RandomForestClassifier", X_train, Y_train, X_val, Y_val, X_test, Y_test)



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