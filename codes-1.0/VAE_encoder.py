import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 1. THE DEFINITION (Solving "Not Defined" Error)
# ==========================================
class SentinelVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=12): 
        super(SentinelVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.05),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(0.05)
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(0.05),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.05),
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
# 2. DATA PREP & MODEL LOADING
# ==========================================
print("--- Loading Data & Weights ---")
df_train = pd.read_csv('master_vae_training_data_final.csv')
df_test = pd.read_csv('Test_1.csv')

features_cols = [c for c in df_train.columns if c not in ['Sample', 'Type', 'Status']]
X_train_raw = df_train[features_cols]
X_test_raw = df_test[features_cols]

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Initialize and Load
vae = SentinelVAE(input_dim=len(features_cols))
vae.load_state_dict(torch.load('sentinel_vae.pth'))
vae.eval()

# ==========================================
# 3. CALCULATING THE HEALTHY THRESHOLD
# ==========================================
with torch.no_grad():
    train_recon, _, _ = vae(torch.FloatTensor(X_train_scaled))
    train_mse = np.mean((X_train_scaled - train_recon.numpy())**2, axis=1)
    # 99th percentile of healthy errors = our "Alarm" line
    threshold = np.percentile(train_mse, 99) 

# ==========================================
# 4. PREDICTING ON TEST VARIATIONS
# ==========================================
with torch.no_grad():
    test_recon, _, _ = vae(torch.FloatTensor(X_test_scaled))
    test_mse = np.mean((X_test_scaled - test_recon.numpy())**2, axis=1)

y_true = df_test['Status'].apply(lambda x: 1 if 'Sick' in str(x) else 0).values
y_pred = (test_mse > threshold).astype(int)

# ==========================================
# 5. FINAL METRICS & CONFUSION MATRIX
# ==========================================
print(f"\n--- SENTINEL RESULTS ---")
print(f"Accuracy:  {accuracy_score(y_true, y_pred)*100:.2f}%")
print(f"F1-Score:  {f1_score(y_true, y_pred)*100:.2f}%")

plt.figure(figsize=(8, 6))
plt.style.use('dark_background')
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Predicted Healthy', 'Predicted Sick'],
            yticklabels=['Actual Healthy', 'Actual Sick'])

plt.title('Sentinel VAE: Final Performance', fontsize=16, pad=20)
plt.savefig('Confusion_Matrix_Final.png', dpi=300)
print("\n✅ Confusion Matrix saved to root folder.")