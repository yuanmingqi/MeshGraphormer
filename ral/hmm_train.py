from ral.hmm import SequenceHandDataset, HandGestureModel, num_classes, num_joints
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torch

data = np.load('./datasets/same/motion_datasets.npz')
sequences, labels = data['sequences'], data['labels']
print(sequences.shape, labels.shape)

# Initialize the model with LSTM parameters
hidden_dim = 64
num_layers = 2
batch_size = 256
learning_rate = 0.001
epochs = 1000

dataset = SequenceHandDataset(sequences, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, dataloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

model = HandGestureModel(num_joints, num_classes, hidden_dim, num_layers)

# Train the model
train_model(model, dataloader, epochs=epochs)
torch.save(model.state_dict(), './models/hmm.pth')