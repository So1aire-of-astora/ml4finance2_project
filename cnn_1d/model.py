import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from typing import Optional

import sys
import os
sys.path.insert(0, os.path.abspath("."))

from utils import get_loader, EarlyStopper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, n_features, n_classes, conv_layers, fc_layers, dropout):
        super(CNN, self).__init__()

        conv_layers_list = []
        in_channels = 1
        current_length = n_features

        for out_channels, kernel_size, stride, padding in conv_layers:
            conv_layers_list.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
            )
            conv_layers_list.append(nn.ReLU())
            current_length = (current_length + 2 * padding - kernel_size) // stride + 1
            in_channels = out_channels

        conv_layers_list.append(nn.MaxPool1d(kernel_size = 2))
        current_length //= 2 
        self.conv_layers = nn.Sequential(*conv_layers_list)

        fc_layers_list = []
        input_size = current_length * in_channels
        for fc_size in fc_layers:
            fc_layers_list.append(nn.Linear(input_size, fc_size))
            fc_layers_list.append(nn.ReLU())
            fc_layers_list.append(nn.Dropout(dropout))
            input_size = fc_size

        fc_layers_list.append(nn.Linear(input_size, n_classes))
        self.fc_layers = nn.Sequential(*fc_layers_list)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        x = self.fc_layers(x)
        return x

def train(model, train_loader, valid_loader, optimizer, criterion, epochs, device, stopper_args: Optional[dict] = None):
    if stopper_args:
        stopper = EarlyStopper(**stopper_args)

    num_batches = len(train_loader)
    num_items = len(train_loader.dataset)
    
    for e in range(epochs):
        model.train()
        total_loss = 0.
        total_correct = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item()

        train_loss = total_loss / num_batches
        train_accuracy = total_correct / num_items
        valid_loss, valid_accuracy, _ = test(model, valid_loader, criterion, device, verbose = 0)
        if not (e+1) % 10:
            print("Epoch %d/%d: Training Loss %.6f\tValidation Loss %.6f\tTraining Accuracy %.2f%%\tValidation Accuracy %.2f%%" %(e+1, epochs, train_loss, valid_loss, train_accuracy*100, valid_accuracy*100))
        if stopper and stopper.early_stop(valid_loss):
            print("[Early stopping]\nEpoch %d/%d: Training Loss %.6f\tValidation Loss %.6f\tTraining Accuracy %.2f%%\tValidation Accuracy %.2f%%" %(e+1, epochs, train_loss, valid_loss, train_accuracy*100, valid_accuracy*100))
            break

def test(model, test_loader, criterion, device, verbose):
    model.eval()

    num_batches = len(test_loader)
    num_items = len(test_loader.dataset)
    
    total_loss = 0.
    total_correct = 0

    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())

    test_loss = total_loss / num_batches
    test_accuracy = total_correct / num_items
    if verbose:
        print("Test Loss %.6f\tTest Accuracy %.2f%%" %(test_loss, test_accuracy*100))
    return test_loss, test_accuracy, all_preds

def main():

    train_feature_path = "./features/feature_train.npy"
    test_feature_path = "./features/feature_test.npy"
    train_label_path = "./features/label_train.csv"
    test_label_path = "./features/label_test.csv"

    batch_size = 32
    valid_size = .2

    train_loader, valid_loader, test_loader, encoder = get_loader(train_feature_path, train_label_path, test_feature_path, test_label_path, batch_size, valid_size)

    n_features = train_loader.dataset[0][0].shape[0]
    n_classes = 4

    conv_config = [
        (16, 3, 1, 1), 
        (32, 3, 1, 1), 
    ]
    fc_config = [128, 64]
    dropout = 0.5

    model = CNN(n_features, n_classes, conv_config, fc_config, dropout).to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 100

    train(model, train_loader, valid_loader, optimizer, criterion, epochs, device, stopper_args = {"threshold": 20, "epsilon": 1e-4})

    test_loss, test_accuracy, pred = test(model, test_loader, criterion, device, verbose = 1)
    pred_labels = encoder.inverse_transform(pred)

    pd.DataFrame(pred_labels, columns = ["Stance"]).to_csv("./temp/preds_cnn.csv", index = False)

if __name__ == "__main__":
    main()