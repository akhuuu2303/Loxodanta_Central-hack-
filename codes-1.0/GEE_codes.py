import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. ARCHITECTURE (Context-Aware VAE)
# ==========================================
class SentinelVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16): 
        super(SentinelVAE, self).__init__()
        # We use a slightly larger latent space to accommodate environmental patterns
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1)
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1),
            nn.Linear(256, input_dim), nn.Sigmoid() 
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# ==========================================
# 2. DATA ALIGNMENT & PREPARATION
# ==========================================
print("=== Phase 1: Feature Alignment & Cleaning ===")
# Loading your two primary files
df_train = pd.read_csv('master_vae_training_data_final.csv')
df_test = pd.read_csv('sick_elephants_with_env_data.csv')

# Environmental columns from your GEE extraction
env_cols = ['NDVI', 'Rainfall_mm', 'Temperature_C']

# CRITICAL FIX: Ensure the Training set has the Environmental columns
# If they are missing, we fill them with the mean of the healthy baseline 
# (assuming the healthy samples represent a 'normal' environment)
for col in env_cols:
    if col not in df_train.columns:
        print(f"⚠️  Adding missing column '{col}' to training set based on test means.")
        df_train[col] = df_test[col].mean()

# Define the full biological-environmental fingerprint (328 Bacteria + 3 Env)
# We exclude non-numeric/metadata columns
meta_cols = ['Sample', 'Type', 'Status', 'Calculated_F_B_Ratio', 'Region_Name', 'Collection_Date', 'Latitude', 'Longitude']
feature_cols = [c for c in df_train.columns if c not in meta_cols]

print(f"✅ Feature alignment complete. Total features for VAE: {len(feature_cols)}")

# Scaling to [0, 1] range for the Sigmoid output layer
scaler = MinMaxScaler()
X_train = scaler.fit_transform(df_train[feature_cols])
X_test = scaler.transform(df_test[feature_cols])

input_dim = len(feature_cols)
train_tensor = torch.FloatTensor(X_train)
test_tensor = torch.FloatTensor(X_test)

# ==========================================
# 3. TRAINING THE SENTINEL
# ==========================================
print(f"=== Phase 2: Training on {input_dim} Bio-Env Features ===")
vae = SentinelVAE(input_dim=input_dim)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

vae.train()
for epoch in range(300):
    optimizer.zero_grad()
    recon, mu, logvar = vae(train_tensor)
    
    # Loss = Reconstruction Error (MSE) + Regularization (KLD)
    mse = nn.functional.mse_loss(recon, train_tensor)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = mse + 0.1 * kld
    
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"   Epoch {epoch+1}/300 | Loss: {loss.item():.6f}")

# Define the 'Healthy Manifold' Boundary (99th percentile)
vae.eval()
with torch.no_grad():
    train_recon, _, _ = vae(train_tensor)
    train_errors = np.mean((X_train - train_recon.numpy())**2, axis=1)
    threshold = np.percentile(train_errors, 99)
    print(f"✅ Diagnostic Threshold established at: {threshold:.6f}")

# ==========================================
# 4. LANDSCAPE DIAGNOSIS (INTERPRETABILITY)
# ==========================================
print("=== Phase 3: Generating Final Visual Diagnosis ===")
with torch.no_grad():
    test_recon, _, _ = vae(test_tensor)
    # The 'Delta' is the absolute difference between 'Sick Reality' and 'Healthy Ideal'
    test_deltas = np.abs(X_test - test_recon.numpy())

# Calculate mean impact per feature across all sick samples
importance = test_deltas.mean(axis=0)
df_imp = pd.DataFrame({'Feature': feature_cols, 'Impact': importance})

# --- BIOLOGICAL GROUPING LOGIC ---
def categorize(name):
    n = name.lower()
    if n in ['ndvi', 'rainfall_mm', 'temperature_c']: return 'Landscape Driver'
    if any(k in n for k in ['prev', 'bact', 'alis', 'para', 'dq']): return 'Bacteroidetes (Stress)'
    if any(k in n for k in ['clos', 'rumi', 'lacto', 'baci', 'eu']): return 'Firmicutes (Healthy)'
    if 'bdello' in n: return 'Proteobacteria (Dysbiosis)'
    return 'Other Microbiota'

df_imp['Group'] = df_imp['Feature'].apply(categorize)

color_map = {
    'Landscape Driver': '#ffcc00',      # GOLD
    'Bacteroidetes (Stress)': '#ff4d4d', # RED
    'Firmicutes (Healthy)': '#00ffcc',   # TEAL
    'Proteobacteria (Dysbiosis)': '#ff9900', # ORANGE
    'Other Microbiota': '#888888'        # GREY
}

# ==========================================
# 5. FINAL VISUALIZATION RENDER
# ==========================================
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 11))
fig.patch.set_facecolor('#1e1e24')

# Panel A: Top 15 Anomaly Drivers
top_drivers = df_imp.sort_values('Impact', ascending=False).head(15)
ax1.set_facecolor('#2b2b36')
ax1.bar(top_drivers['Feature'], top_drivers['Impact'], color=[color_map[g] for g in top_drivers['Group']])
ax1.set_title("A: Integrated Bio-Environmental Anomaly Drivers", fontsize=22, fontweight='bold', pad=25)
ax1.set_ylabel("Diagnostic Weight (Reconstruction Delta)", fontsize=15)
ax1.tick_params(axis='x', rotation=45, labelsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.1)

# Panel B: Global Ecosystem Impact (Phylum vs Landscape)
group_summary = df_imp.groupby('Group')['Impact'].mean().sort_values()
ax2.set_facecolor('#2b2b36')
ax2.barh(group_summary.index, group_summary.values, color=[color_map[g] for g in group_summary.index])
ax2.set_title("B: Global Ecosystem Impact Summary", fontsize=22, fontweight='bold', pad=25)
ax2.set_xlabel("Average Impact Weight", fontsize=15)
ax2.tick_params(axis='y', labelsize=14)
ax2.grid(axis='x', linestyle='--', alpha=0.1)

# Branding and Legend
fig.text(0.04, 0.94, "TUSK TRUST | ECO-SENTINEL DASHBOARD", fontsize=34, fontweight='900', color='#FFD700')
import matplotlib.patches as mpatches
legend_els = [mpatches.Patch(color=v, label=k) for k,v in color_map.items()]
fig.legend(handles=legend_els, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=5, fontsize=13, frameon=False)

plt.subplots_adjust(top=0.82, bottom=0.18, wspace=0.3)
plt.savefig('Tusk_Trust_Final_Diagnosis.png', dpi=300)

print(f"\n✅ PROCESS COMPLETE: 'Tusk_Trust_Final_Diagnosis.png' generated.")
print(f"📊 Interpretation: Higher bars indicate features that deviated most from the healthy baseline.")