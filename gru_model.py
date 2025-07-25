
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(GRUAutoencoder, self).__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, hidden = self.encoder(x)
        hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)  # Repeat for each time step
        decoded, _ = self.decoder(hidden_repeated)
        out = self.output_layer(decoded)
        return out

def gru_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum')

def train_and_predict(train_data, test_data, input_dim, epochs=100, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUAutoencoder(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x = model(x)
            loss = gru_loss(recon_x, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(test_data, dtype=torch.float32).to(device)
        pred = model(x_test)
    return pred.cpu().numpy()
