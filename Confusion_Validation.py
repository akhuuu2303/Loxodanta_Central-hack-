import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("=== 1. LOADING TEST RESULTS ===")
try:
    df = pd.read_csv('Test_1_Dashboard_Results.csv')
    print(f"Successfully loaded {len(df)} predictions.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not find 'Test_1_Dashboard_Results.csv'. Make sure you ran the previous script! Error: {e}")
    exit()

# === 2. DEFINE GROUND TRUTH VS AI PREDICTION ===
# Ground Truth: 1 if the elephant is actually Sick, 0 if Healthy
y_true = df['Status'].apply(lambda x: 1 if x == 'Sick' else 0)

# AI Prediction: 0 if AI called it GREEN (Safe). 
# We classify both YELLOW (Warning) and RED (Critical) as a positive 'Anomaly' flag (1)
y_pred = df['Zone'].apply(lambda x: 0 if 'GREEN' in str(x) else 1)

# === 3. CALCULATE CORE METRICS ===
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n" + "="*45)
print("🏆 SENTINEL AI - FINAL PERFORMANCE METRICS 🏆")
print("="*45)
print(f"✅ Accuracy:  {accuracy * 100:.2f}% (Total correct predictions)")
print(f"✅ Precision: {precision * 100:.2f}% (When AI said 'Sick', was it right?)")
print(f"✅ Recall:    {recall * 100:.2f}% (Out of all sick elephants, how many did it catch?)")
print(f"✅ F1-Score:  {f1 * 100:.2f}% (The ultimate harmonic balance of the model)")
print("="*45)

# === 4. GENERATE PRESENTATION-READY CONFUSION MATRIX ===
plt.figure(figsize=(8, 6))

# Use a sleek dark background suitable for a tech/AI pitch deck
plt.style.use('dark_background')

# Create the heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            annot_kws={"size": 24, "weight": "bold"},
            xticklabels=['Predicted Healthy', 'Predicted Sick'],
            yticklabels=['Actual Healthy', 'Actual Sick'],
            linewidths=2, linecolor='black')

# Add titles and labels
plt.title('Sentinel VAE: Final Confusion Matrix\n(Unseen African Elephant Data)', 
          fontsize=16, pad=20, color='white', weight='bold')
plt.yticks(rotation=0, fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()

# Save the plot
output_image = 'Final_Pitch_Confusion_Matrix.png'
plt.savefig(output_image, dpi=300, transparent=True)

print(f"\n📸 SUCCESS: Saved high-resolution graphic to -> '{output_image}'")
print("Drop this image straight into your presentation slides!")