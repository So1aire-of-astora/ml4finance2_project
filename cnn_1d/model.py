import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from data import get_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, n_features, num_classes, conv_layers, fc_layers, dropout_prob=0.5):
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

        conv_layers_list.append(nn.MaxPool1d(kernel_size=2))
        current_length //= 2 
        self.conv_layers = nn.Sequential(*conv_layers_list)

        fc_layers_list = []
        input_size = current_length * in_channels
        for fc_size in fc_layers:
            fc_layers_list.append(nn.Linear(input_size, fc_size))
            fc_layers_list.append(nn.ReLU())
            fc_layers_list.append(nn.Dropout(dropout_prob))
            input_size = fc_size

        # Output layer
        fc_layers_list.append(nn.Linear(input_size, num_classes))
        self.fc_layers = nn.Sequential(*fc_layers_list)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        x = self.fc_layers(x)
        return x

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions
    return epoch_loss, epoch_accuracy

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(test_loader)
    epoch_accuracy = correct_predictions / total_predictions

    return epoch_loss, epoch_accuracy, all_preds, all_labels

def main():

    train_feature_path = "./features/feature_train.npy"
    test_feature_path = "./features/feature_test.npy"
    train_label_path = "./features/label_train.csv"
    test_label_path = "./features/label_test.csv"

    batch_size = 32

    train_loader, valid_loader, test_loader = get_loader(train_feature_path, train_label_path, test_feature_path, test_label_path, batch_size)

    n_features = train_loader.dataset[0][0].shape[0]

    num_classes = 4 

    conv_config = [
        (16, 3, 1, 1), 
        (32, 3, 1, 1), 
    ]
    fc_config = [128, 64] 
    dropout_prob = 0.5

    model = CNN(n_features, num_classes, conv_config, fc_config, dropout_prob).to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_accuracy, test_preds, test_labels = test(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

if __name__ == "__main__":
    main()