import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=== LOADING AI RESULTS ===")
df = pd.read_csv('Test_1_Dashboard_Results.csv')

# THE FIX: Check if the word 'Sick' is anywhere in the Status!
y_true = df['Status'].apply(lambda x: 1 if 'Sick' in str(x) else 0)

# AI Prediction: 0 if GREEN, 1 if YELLOW or RED
y_pred = df['Zone'].apply(lambda x: 0 if 'GREEN' in str(x) else 1)

# Recalculate everything
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n" + "="*50)
print("🏆 SENTINEL AI - TRUE PERFORMANCE METRICS 🏆")
print("="*50)
print(f"✅ Accuracy:  {accuracy * 100:.2f}%")
print(f"✅ Precision: {precision * 100:.2f}%")
print(f"✅ Recall:    {recall * 100:.2f}%")
print(f"✅ F1-Score:  {f1 * 100:.2f}%")
print("="*50)

# Generate the Confusion Matrix
plt.figure(figsize=(8, 6))
plt.style.use('dark_background')
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            annot_kws={"size": 24, "weight": "bold"},
            xticklabels=['Predicted Healthy', 'Predicted Sick'],
            yticklabels=['Actual Healthy', 'Actual Sick'],
            linewidths=2, linecolor='black')

plt.title('Sentinel VAE: True Confusion Matrix', fontsize=16, pad=20, color='white', weight='bold')
plt.yticks(rotation=0, fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.savefig('Final_Pitch_Confusion_Matrix.png', dpi=300, transparent=True)
print("\n📸 SUCCESS: Saved graphic -> 'Final_Pitch_Confusion_Matrix.png'")