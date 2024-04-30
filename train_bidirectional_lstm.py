import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from preprocess import generate_training_sequences, SEQUENCE_LENGTH

OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS_FUNCTION = nn.CrossEntropyLoss()
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model_new.pt"



if torch.cuda.is_available():
    device = torch.device("cuda:0")  
    print("Training on GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Training on CPU")



class MusicDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


class LSTMModel(nn.Module):
    def __init__(self, output_units, num_units):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(output_units, num_units[0], batch_first = True, bidirectional = True)
        self.dropout = nn.Dropout(0.2)
        #self.linear = nn.Linear(num_units[0], output_units)
        self.linear = nn.Linear(num_units[0] * 2, output_units)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.dropout(output[:, -1, :])
        output = self.linear(output)
        return output
        
        
def build_model(output_units, num_units, loss_function, learning_rate):
    model = LSTMModel(output_units, num_units)
    model = model.to(device)
    loss_fn = loss_function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, loss_fn, optimizer


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss_function=LOSS_FUNCTION, learning_rate=LEARNING_RATE):
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    #inputs = torch.tensor(inputs, dtype=torch.long)
    #targets = torch.tensor(targets, dtype=torch.long)
    
    inputs = inputs.clone().detach().requires_grad_(False)
    targets = targets.clone().detach().requires_grad_(False)
    
    dataset = MusicDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model, loss_fn, optimizer = build_model(output_units, num_units, loss_function, learning_rate)

    for epoch in range(EPOCHS):
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device).float()  # Ensure float input
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
    
    
    
