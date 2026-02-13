import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("--- 1. LOADING HEALTHY AFRICAN BASELINE ---")
try:
    master_df = pd.read_csv('master_vae_training_data_final.csv')
except Exception as e:
    print(f"CRITICAL ERROR: Could not load master_vae_training_data_final.csv. {e}")
    exit()

# Drop the Asian elephants (Wild_2) to ensure a strictly African baseline
healthy_df = master_df[~master_df['Type'].str.contains('Wild_2')].copy()

# Randomly sample 20 healthy elephants
test_healthy = healthy_df.sample(n=20, random_state=42).copy()
test_healthy['Status'] = 'Healthy'

print("--- 2. LOADING SIMULATED SICK ELEPHANTS ---")
try:
    sick_df = pd.read_excel('Simulated_Sick_Elephants_Realistic_Spike.xlsx')
except Exception as e:
    print(f"CRITICAL ERROR: Could not load Simulated_Sick_Elephants_Realistic_Spike.xlsx. {e}")
    exit()

# Randomly sample 20 sick elephants
test_sick = sick_df.sample(n=20, random_state=42).copy()

# We need to drop the 'Calculated_F_B_Ratio' column from the sick dataset 
# so the columns perfectly match the 328 bacterial features the VAE expects.
if 'Calculated_F_B_Ratio' in test_sick.columns:
    test_sick = test_sick.drop(columns=['Calculated_F_B_Ratio'])

print("--- 3. MERGING INTO Test_1.csv ---")
# Combine them into a single 40-row dataset
test_1_df = pd.concat([test_healthy, test_sick], ignore_index=True)

# Reorder columns to make it clean (Metadata first, then Bacteria)
features_cols = [c for c in test_1_df.columns if c not in ['Sample', 'Type', 'Status']]
final_cols = ['Sample', 'Type', 'Status'] + features_cols
test_1_df = test_1_df[final_cols]

# Shuffle the dataset so Healthy and Sick are mixed up
test_1_df = test_1_df.sample(frac=1, random_state=99).reset_index(drop=True)

# Save to CSV
output_name = 'Test_1.csv'
test_1_df.to_csv(output_name, index=False)

print("\n" + "="*50)
print(f"SUCCESS: {output_name} created!")
print(f"Total Samples: {len(test_1_df)}")
print("\nClass Distribution:")
print(test_1_df['Status'].value_counts().to_string())
print("\nPreview (First 5 Rows):")
print(test_1_df[['Sample', 'Type', 'Status']].head(5))
print("="*50)