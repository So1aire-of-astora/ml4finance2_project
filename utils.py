import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder

class NewsDataset(Dataset):
    def __init__(self, feature_path, label_path, label_col):

        self.features = np.load(feature_path)
        self.labels = pd.read_csv(label_path)[label_col]
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        return feature, label

def get_loader(train_feature_path, train_label_path, test_feature_path, test_label_path, batch_size, valid_size, col = "Stance"):
    train_valid_data = NewsDataset(train_feature_path, train_label_path, col)
    test_data = NewsDataset(test_feature_path, test_label_path, col)
    train_data, valid_data = random_split(train_valid_data, [1 - valid_size, valid_size])
    return DataLoader(train_data, batch_size = batch_size, shuffle = True), DataLoader(valid_data, batch_size = batch_size, shuffle = True), DataLoader(test_data, batch_size=batch_size, shuffle = False), \
            test_data.label_encoder

class EarlyStopper:
    def __init__(self, threshold = float("inf"), epsilon = float("inf")):
        self.threshold = threshold
        self.epsilon = epsilon
        self.counter = 0
        self.min_valid_loss = float('inf')

    def early_stop(self, valid_loss):
        if valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss
            self.counter = 0
        elif valid_loss > (self.min_valid_loss + self.epsilon):
            self.counter += 1
            if self.counter >= self.threshold:
                return True
        return False
