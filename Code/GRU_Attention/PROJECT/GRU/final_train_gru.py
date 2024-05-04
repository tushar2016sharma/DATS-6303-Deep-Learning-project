import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from final_preprocess_gru import generate_training_sequences, SEQUENCE_LENGTH
from tqdm import tqdm
import json

import torch.nn.functional as F

# Constants
NUM_UNITS = 128
BATCH_SIZE = 100
LOSS = nn.CrossEntropyLoss()
LEARNING_RATE = 0.0005
EPOCHS = 8
SAVE_MODEL_PATH = "model.pth"
with open('mapping.json', 'r') as file:
    data = json.load(file)
    size = len(data)  # If the JSON is an object, this returns the number of top-level keys.

print("Number of top-level keys:", size)
OUTPUT_UNITS = size
INPUT_UNITS = size


class AdvancedAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AdvancedAttention, self).__init__()
        self.attention_fc = nn.Linear(input_dim, attention_dim)
        self.value_fc = nn.Linear(input_dim, attention_dim)
        self.query_fc = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        attention_scores = self.query_fc(torch.tanh(self.attention_fc(x) + self.value_fc(x)))
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_sum = torch.sum(x * attention_weights, dim=1)
        return weighted_sum
class MusicModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4):
        super(MusicModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # GRU layer, bidirectional
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=0.5, bidirectional=True)

        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # Enhanced Attention
        self.attention = AdvancedAttention(hidden_size * 2, hidden_size)

        # More Dense layers
        self.dense1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dense3 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.6)
    def forward(self, x):
        # GRU forward pass
        x, _ = self.gru(x)

        # Applying enhanced attention
        x = self.attention(x)

        # Applying batch normalization
        x = self.batch_norm(x)

        # Passing through multiple Dense layers
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dense3(x)

        return x

def train_model():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Model setup
    model = MusicModel(OUTPUT_UNITS, NUM_UNITS, OUTPUT_UNITS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    # Training loop
    for epoch in range(EPOCHS):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        # Initialize tqdm progress bar
        data_generator = generate_training_sequences(SEQUENCE_LENGTH, BATCH_SIZE)
        progress_bar = tqdm(enumerate(data_generator), desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (x_batch, y_batch) in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)

            # Calculate loss
            loss = LOSS(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy (assuming classification task)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == y_batch).sum().item()
            total_samples += y_batch.size(0)

            # Update progress bar with loss info
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / (batch_idx + 1)
        accuracy = total_correct / total_samples * 100  # Calculate accuracy as percentage

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print("Model saved to", SAVE_MODEL_PATH)
if __name__ == "__main__":
    train_model()