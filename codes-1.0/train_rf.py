import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1. Recreate the exact VAE architecture
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

print("Loading Master Data & VAE...")
master_df = pd.read_csv('master_vae_training_data_final.csv')

# Safely filter out any potential metadata columns
meta_cols = ['Sample', 'Type', 'Status', 'Condition', 'Label', 'Health', 'Calculated_F_B_Ratio', 'Region_Name', 'Collection_Date', 'Latitude', 'Longitude']
feature_cols = [c for c in master_df.columns if c not in meta_cols][:326]

# Pad missing environmental columns with 0
for env in ['NDVI', 'Rainfall_mm', 'Temperature_C']:
    if env not in master_df.columns: master_df[env] = 0.0

# Scale data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(master_df[feature_cols])

# Load VAE weights
vae = SentinelVAE()
vae.load_state_dict(torch.load('sentinel_vae.pth', map_location='cpu', weights_only=True), strict=False)
vae.eval()

# Generate Anomaly Scores
with torch.no_grad():
    recon, _, _ = vae(torch.FloatTensor(X_scaled))
    train_errors = np.mean((X_scaled - recon.numpy())**2, axis=1)

print("Training Unified Random Forest...")
# 2. CREATE THE HYBRID DATASET
rf_X = pd.DataFrame({
    'VAE_Anomaly_Score': train_errors,
    'NDVI': master_df['NDVI'],
    'Rainfall_mm': master_df['Rainfall_mm'],
    'Temperature_C': master_df['Temperature_C']
})

# 3. DYNAMIC LABELING (The Fix)
status_col = next((col for col in master_df.columns if col.lower() in ['status', 'condition', 'label', 'health']), None)

if status_col:
    print(f"✅ Found label column: '{status_col}'")
    rf_y = master_df[status_col].apply(lambda x: 'At Risk' if 'Sick' in str(x) or 'Stressed' in str(x) or str(x) == '1' else 'Healthy')
else:
    print("⚠️ No 'Status' column found. Using VAE self-supervision to generate training labels...")
    # Use the 95th percentile so the RF has enough "At Risk" examples (top 5%) to learn the boundary
    threshold = np.percentile(train_errors, 95)
    rf_y = ['At Risk' if e > threshold else 'Healthy' for e in train_errors]

# 4. Train and Save
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(rf_X, rf_y)

joblib.dump(rf_model, 'sentinel_rf.pkl')
print("✅ SUCCESS! 'sentinel_rf.pkl' has been generated.")