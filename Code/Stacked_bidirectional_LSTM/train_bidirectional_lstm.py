# -*- coding: Windows-1252 -*-

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from preprocess_erk import generate_training_sequences, SEQUENCE_LENGTH

# Constants
NUM_UNITS = [256, 256]
LOSS_FUNCTION = nn.CrossEntropyLoss()
LEARNING_RATE = 0.0005
EPOCHS = 100
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model_bidirectional_LSTM.pt"

# Checking the device type
if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print("Training on GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Training on CPU")
    
# Load the json file contaning the unique symbols    
with open('mapping.json', 'r') as file:
    data = json.load(file)
    size = len(data)

# flexible output size based on unique symbols found in train data
OUTPUT_UNITS = size   

# Class for dataset
class MusicDataset(Dataset):
    def __init__(self, inputs, targets): 
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)  

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index] 


# Bidirectional stacked LSTM Model architecture
class LSTMModel(nn.Module):
  def __init__(self, output_units, num_units):
    super(LSTMModel, self).__init__()
    self.lstm1 = nn.LSTM(output_units, num_units[0], batch_first=True, bidirectional=True)
    self.dropout1 = nn.Dropout(0.2)

    self.lstm2 = nn.LSTM(num_units[0] * 2, num_units[1], batch_first=True, bidirectional=True)  # Second LSTM layer
    self.dropout2 = nn.Dropout(0.2)

    self.linear = nn.Linear(num_units[1] * 2, output_units)  # Output layer 

  # Forward pass
  def forward(self, x):
    output, _ = self.lstm1(x)
    output = self.dropout1(output)

    output, _ = self.lstm2(output)  # Pass output of first LSTM through the second
    output = self.dropout2(output)

    output = output[:, -1, :]  # Take the last timestep output
    output = self.linear(output)
    return output

        
# Build model        
def build_model(output_units, num_units, loss_function, learning_rate):
    model = LSTMModel(output_units, num_units)
    model = model.to(device)
    loss_fn = loss_function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_fn, optimizer

# Training loop
def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss_function=LOSS_FUNCTION, learning_rate=LEARNING_RATE):

    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    inputs = inputs.clone().detach().requires_grad_(False)
    targets = targets.clone().detach().requires_grad_(False)
    
    dataset = MusicDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model, loss_fn, optimizer = build_model(output_units, num_units, loss_function, learning_rate)

    for epoch in range(EPOCHS):
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device).float() 
            batch_y = batch_y.to(device) 
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()
    
    
    
