import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from preprocess import generate_training_sequences, SEQUENCE_LENGTH
from tqdm import tqdm
import torch.nn.functional as F


# Constants
OUTPUT_UNITS = 38
NUM_UNITS = 128
BATCH_SIZE = 100
LOSS = nn.CrossEntropyLoss()
LEARNING_RATE = 0.001
EPOCHS = 20
SAVE_MODEL_PATH = "model.pth"


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

class MusicModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(MusicModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.3)
        self.attention = MultiHeadAttention(hidden_size * 2, 8)
        self.dense = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(self.layer_norm(x))
        x = self.attention(x, x, x)
        x = self.dense(x[:, -1, :])  # Assuming we still only want the final timestep for prediction
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
    # Training loop
    for epoch in range(EPOCHS):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        # Initialize tqdm progress bar
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{EPOCHS}")

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

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples * 100  # Calculate accuracy as percentage

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Save the model
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print("Model saved to", SAVE_MODEL_PATH)



if __name__ == "__main__":
    train_model()
