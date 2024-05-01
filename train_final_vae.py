import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from preprocess import generate_training_sequences, SEQUENCE_LENGTH
import json
import sys
import time
from tqdm import tqdm  

LATENT_DIM = 50
HIDDEN_DIM = 256
BATCH_SIZE = 32
LEARNING_RATE = 0.005
EPOCHS = 10
SAVE_MODEL_PATH = "vae_model.pth"


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

with open('mapping.json', 'r') as file:
    data = json.load(file)
    input_dim = len(data)  

print("Number of top-level keys:", input_dim)
OUTPUT_UNITS = input_dim 

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(VAE, self).__init__()
        # Encoder part
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  
        self.relu2 = nn.ReLU()
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

        # Decoder part
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)  
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_dim, output_dim)

    def encode(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.relu2(self.fc2(x))
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

    def decode(self, z):
        z = self.relu3(self.fc3(z))
        z = self.dropout2(self.relu4(self.fc4(z)))
        return torch.sigmoid(self.fc5(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=0.5):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum') / x.numel()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

def train_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim, HIDDEN_DIM, LATENT_DIM, input_dim)
    model.apply(init_weights)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0
        average_batch_time = 0  

        batch_generator = generate_training_sequences(SEQUENCE_LENGTH, BATCH_SIZE, device)
        progress_bar = tqdm(enumerate(batch_generator), desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)

        best_loss = float('inf')

        for batch_idx, (inputs, targets, _) in progress_bar:
            batch_start_time = time.time()
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)
            loss = loss_function(recon_batch, inputs, mu, logvar)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            current_batch_time = time.time() - batch_start_time
            average_batch_time = ((average_batch_time * (num_batches - 1)) + current_batch_time) / num_batches

            progress_bar.set_postfix(loss=f"{loss.item():}")

        progress_bar.close()  
        average_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{EPOCHS} completed: Average Loss = {average_loss:}")

        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print("Model saved to", SAVE_MODEL_PATH)

if __name__ == "__main__":
    train_vae()
