from hmm import SequenceHandDataset, HandGestureModel, num_classes, num_joints
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import random

person = 'ymq'
data = np.load(f'../datasets/{person}/processed_data.npz')
sequences, labels = data['sequences'], data['labels']
print(sequences.shape, labels.shape)

# 检查是否支持 MPS
device = torch.device("cpu")
print("Using device:", device)

# 拆分数据集
sequences_train, sequences_val, labels_train, labels_val = train_test_split(
    sequences, labels, test_size=0.2, stratify=labels, random_state=42
)

seed = 4     # Set seed for reproducibility

np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)
random.seed(seed)

# Initialize the model with LSTM parameters
hidden_dim = 64
num_layers = 2
batch_size = 512
learning_rate = 0.00025
epochs = 100

# 创建 DataLoader
train_dataset = SequenceHandDataset(sequences_train, labels_train)
val_dataset = SequenceHandDataset(sequences_val, labels_val)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def validate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            all_predictions.extend(predicted.tolist())
            all_targets.extend(target.tolist())
            correct += (predicted == target).sum().item()
            total += target.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    return avg_loss, accuracy, f1

train_losses = []
val_losses = []
val_accuracies = []
val_f1_scores = []

def train_model(model, train_dataloader, val_dataloader, epochs, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = total_loss / len(train_dataloader)
        train_losses.append(train_loss)
        val_loss, val_accuracy, val_f1 = validate_model(model, val_dataloader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}, Val F1-score: {val_f1}')

model = HandGestureModel(num_joints, num_classes, hidden_dim, num_layers)

# Train the model
train_model(model, train_dataloader, val_dataloader, epochs, device)

# Save the model and training metrics
torch.save({
    'model_state_dict': model.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'val_accuracies': val_accuracies,
    'val_f1_scores': val_f1_scores
}, f'../logs/hmm_{person}_s{seed}.pth')

print("Model and metrics saved successfully.")
