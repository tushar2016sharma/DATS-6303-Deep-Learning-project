import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from preprocess import generate_training_sequences, SEQUENCE_LENGTH

# Constants
OUTPUT_UNITS = 38
NUM_UNITS = 128
BATCH_SIZE = 32
LOSS = nn.CrossEntropyLoss()
LEARNING_RATE = 0.001
EPOCHS = 13
SAVE_MODEL_PATH = "model.pth"


class MusicModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=25):
        super(MusicModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.dropout(x[:, -1, :])  # Only use the output at the last time step
        x = self.dense(x)
        return x


def train_model():
    # Load data
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    inputs = inputs.clone().detach().requires_grad_(False).float()
    targets = targets.clone().detach().requires_grad_(False).long()

    # Create Dataset and DataLoader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Model
    model = MusicModel(OUTPUT_UNITS, NUM_UNITS, OUTPUT_UNITS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    # Training loop
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = LOSS(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print("Model saved to", SAVE_MODEL_PATH)


if __name__ == "__main__":
    train_model()
