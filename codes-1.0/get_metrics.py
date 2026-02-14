import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, precision_score

# 1. Load your VAE architecture here (the SentinelVAE class)
class SentinelVAE(nn.Module):
    def __init__(self, input_dim=326, latent_dim=12): 
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Identity(), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1))
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Identity(), nn.Linear(128, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Identity(), nn.Linear(256, input_dim), nn.Sigmoid())
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.randn_like(torch.exp(0.5 * logvar)) * torch.exp(0.5 * logvar)
        return self.decoder(z), mu, logvar

# 2. Load the Data and Model
master_df = pd.read_csv('master_vae_training_data_final.csv')
meta_cols = ['Sample', 'Type', 'Status', 'Condition', 'Label', 'Health', 'Calculated_F_B_Ratio', 'Region_Name', 'Collection_Date', 'Latitude', 'Longitude']
feature_cols = [c for c in master_df.columns if c not in meta_cols][:326]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(master_df[feature_cols])

vae = SentinelVAE()
vae.load_state_dict(torch.load('sentinel_vae.pth', map_location='cpu', weights_only=True), strict=False)
vae.eval()

# 3. Calculate Errors
with torch.no_grad():
    recon, _, _ = vae(torch.FloatTensor(X_scaled))
    errors = np.mean((X_scaled - recon.numpy())**2, axis=1)

# 4. Generate Predictions vs Reality
threshold = np.percentile(errors, 75) # Using the 75th percentile we set earlier
y_pred = [1 if e > threshold else 0 for e in errors]

# Find the real labels
status_col = next((col for col in master_df.columns if col.lower() in ['status', 'condition', 'label']), None)
danger_words = ['sick', 'stress', 'risk', 'disease', '1', 'spike']

if status_col:
    y_true = master_df[status_col].astype(str).str.lower().apply(lambda x: 1 if any(w in x for w in danger_words) else 0)
    
    # CALCULATE THE REAL MATH
    real_recall = recall_score(y_true, y_pred, zero_division=0) * 100
    real_precision = precision_score(y_true, y_pred, zero_division=0) * 100
    
    print(f"YOUR TRUE RECALL: {real_recall:.1f}%")
    print(f"YOUR TRUE PRECISION: {real_precision:.1f}%")
else:
    print("Could not find ground truth labels to calculate exact metrics.")