import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 1. LOAD THE DATA
print("--- 1. LOADING MASTER DATASET ---")
try:
    df = pd.read_csv('master_vae_training_data_final.csv')
except Exception as e:
    print(f"CRITICAL ERROR: Could not load data. {e}")
    exit()

# DROP THE ASIAN ELEPHANTS (Wild_2) to keep it strictly African
df = df[~df['Type'].str.contains('Wild_2')].copy()

meta_cols = ['Sample', 'Type']
features = df.drop(columns=meta_cols)

# 2. SYNTHETIC DYSBIOSIS INJECTION (Simulating Sick African Elephants)
print("--- 2. GENERATING SIMULATED 'SICK/STRESSED' AFRICAN ELEPHANTS ---")
firmicutes_genera = ['Clostridium', 'Ruminococcus', 'Coprococcus', 'Oscillospira', 'Roseburia', 'Lactobacillus', 'Streptococcus', 'Bacillus', 'Eubacterium', 'Blautia', 'Faecalibacterium']
bacteroidetes_genera = ['Bacteroides', 'Prevotella', 'Paludibacter', 'Parabacteroides', 'Alistipes', 'Porphyromonas']

def match_phylum(col_name, phylum_list):
    return any(g.lower() in col_name.lower() for g in phylum_list)

f_cols = [c for c in features.columns if match_phylum(c, firmicutes_genera)]
b_cols = [c for c in features.columns if match_phylum(c, bacteroidetes_genera)]

# Grab the healthy wild African elephants to act as the base
wild_base = df[df['Type'] == 'Wild_1'].copy()
simulated_sick_list = []

for i, row in wild_base.iterrows():
    sick_row = row.copy()
    sick_row['Sample'] = f"Sick_Simulated_{row['Sample']}"
    sick_row['Type'] = 'Simulated_Stressed'
    
    # INJECT THE DYSBIOSIS: Crush Firmicutes, Spike Bacteroidetes
    for col in f_cols:
        sick_row[col] = float(sick_row[col]) * 0.10  # Drop to 10% of normal
    for col in b_cols:
        sick_row[col] = float(sick_row[col]) * 4.0   # Increase by 400%
        
    # Add random noise so they aren't perfect clones
    feature_vals = sick_row[features.columns].values.astype(float)
    noise = np.random.normal(0, 0.05, size=feature_vals.shape)
    new_vals = np.clip(feature_vals + noise, 0, None)
    if new_vals.sum() > 0:
        new_vals = (new_vals / new_vals.sum()) * 100.0 # Re-normalize to 100%
        
    sick_row[features.columns] = new_vals
    simulated_sick_list.append(sick_row)

# Add the simulated sick elephants to our main dataframe
df_sick = pd.DataFrame(simulated_sick_list)
df = pd.concat([df, df_sick], ignore_index=True)

# Recalculate features and metadata
metadata = df[meta_cols]
features = df.drop(columns=meta_cols)

# 3. CALCULATE F/B RATIO FOR THE DASHBOARD
f_sum = features[f_cols].sum(axis=1) + 1e-5
b_sum = features[b_cols].sum(axis=1) + 1e-5
fb_ratio = f_sum / b_sum

# 4. PERFORM TRAIN/TEST SPLIT
print("--- 3. PERFORMING TRAIN/TEST SPLIT ---")
# TRAIN ONLY ON HEALTHY BASELINE (Real & Synthetic AMDB and Wild_1)
train_mask = df['Type'].isin(['AMDB', 'Syn_AMDB', 'Wild_1', 'Syn_Wild_1'])
train_features = features[train_mask]

print(f"   -> Training Set (Healthy African Baseline): {len(train_features)} samples.")
print(f"   -> Testing Set (Simulated Stressed African): {len(df) - len(train_features)} samples.")

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_features.values)
X_all_scaled = scaler.transform(features.values)

train_tensor = torch.FloatTensor(X_train_scaled)
all_tensor = torch.FloatTensor(X_all_scaled)

# 5. VAE ARCHITECTURE
class SentinelVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=6): 
        super(SentinelVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2), nn.Dropout(0.15)
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2), nn.Dropout(0.15),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.Dropout(0.15),
            nn.Linear(128, input_dim), nn.Sigmoid() 
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

input_dim = train_tensor.shape[1]
vae = SentinelVAE(input_dim=input_dim)
optimizer = optim.Adam(vae.parameters(), lr=0.001, weight_decay=1e-5) 

def loss_function(recon_x, x, mu, logvar, beta):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + (beta * KLD)

# 6. TRAINING LOOP
print(f"\n--- 4. TRAINING VAE EXCLUSIVELY ON HEALTHY AFRICAN BASELINE ---")
epochs = 600
vae.train() 
for epoch in range(epochs):
    optimizer.zero_grad()
    recon_batch, mu, logvar = vae(train_tensor)
    current_beta = min(0.5, epoch / 400.0) 
    loss = loss_function(recon_batch, train_tensor, mu, logvar, current_beta)
    loss.backward()
    optimizer.step()

# 7. EVALUATION & RISK DASHBOARD
print("\n--- 5. GENERATING FINAL PRESENTATION DASHBOARD ---")
vae.eval() 
with torch.no_grad():
    reconstructed_all, _, _ = vae(all_tensor)
    mse_per_sample = np.mean(np.power(all_tensor.numpy() - reconstructed_all.numpy(), 2), axis=1)

results = metadata.copy()
results['F/B_Ratio'] = fb_ratio.round(3)
results['Risk_Raw'] = mse_per_sample

def calibrate_score(mse_value):
    max_expected_mse = 0.15 
    score = (mse_value / max_expected_mse) * 100
    return min(100.0, max(0.0, score))

results['Risk_Score'] = results['Risk_Raw'].apply(calibrate_score)

def get_risk_category(score):
    if score < 30: return "Safe (Healthy Baseline)"
    elif score < 70: return "Warning (Monitor)"
    else: return "CRITICAL ANOMALY"

results['Status'] = results['Risk_Score'].apply(get_risk_category)

# Save and Sort
results = results.sort_values(by='Risk_Score', ascending=False)
results.to_csv('final_presentation_dashboard.csv', index=False)

print("\n" + "="*60)
print("SUCCESS: 'final_presentation_dashboard.csv' created!")
print("="*60)
print("TOP ANOMALIES (Should be your new Simulated Sick African Elephants):")
print(results[['Sample', 'Type', 'Risk_Score', 'F/B_Ratio']].head(7))
print("="*60)
print("BOTTOM HEALTHY ELEPHANTS:")
print(results[['Sample', 'Type', 'Risk_Score', 'F/B_Ratio']].tail(5))
print("="*60)