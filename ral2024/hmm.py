import torch
import torch.nn as nn
from torch.utils.data import Dataset

num_joints = 21
motion_labels = ['keep', 'come', 'back', 'stop', 'ring']
num_classes = len(motion_labels)

class HandGestureModel(nn.Module):
    def __init__(self, num_joints, num_classes, hidden_dim, num_layers):
        super(HandGestureModel, self).__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=num_joints * 3, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        # x should be of shape (batch, sequence, features)
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return out

class SequenceHandDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # sequences[idx] should be of shape (sequence_length, num_joints * 3)
        return torch.tensor(self.sequences[idx], dtype=torch.float32), self.labels[idx]
