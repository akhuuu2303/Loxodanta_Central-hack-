import pandas as pd
import numpy as np
import os

# --- EXACT PATHS FROM YOUR SCAN ---
amdb_folder = 'processed_amdb_genus'
wild1_file = 'wild_supp_data.xls'
wild2_file = 'wild_supp_data2.xls'
output_filename = 'master_vae_training_data_final.csv'
target_total_samples = 300
target_per_group = target_total_samples // 2  # 150 each

def build_balanced_master_dataset_final():
    print(f"Current Directory: {os.getcwd()}")
    all_real_data = []
    n_count = 1
    w_count = 1
    
    # --- 1. AMDB DATA ---
    print("\n--- 1. LOADING AMDB DATA (N_ series) ---")
    for file in os.listdir(amdb_folder):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(amdb_folder, file))
                col_map = {c: 'Genus' if 'genus' in c.lower() or 'taxonomy' in c.lower() else 'Abundance' if 'abundance' in c.lower() else c for c in df.columns}
                df = df.rename(columns=col_map)
                
                if 'Genus' in df.columns and 'Abundance' in df.columns:
                    df['Genus'] = df['Genus'].astype(str).str.replace('g__', '')
                    df = df.groupby('Genus')['Abundance'].sum().reset_index()
                    df['Sample'] = f"N_{n_count}"
                    df['Type'] = 'AMDB'
                    all_real_data.append(df)
                    n_count += 1
            except Exception as e:
                print(f"   [!] Error on {file}: {e}")
    print(f"   => Loaded {n_count - 1} Real AMDB Samples.")

    # --- 2. WILD DATA 1 ---
    print(f"\n--- 2. LOADING WILD DATA 1 ({wild1_file}) ---")
    try:
        # Try reading as CSV first, if it fails, read as Excel and pull the specific sheet
        try:
            df_w1 = pd.read_csv(wild1_file, skiprows=1)
        except:
            try:
                df_w1 = pd.read_excel(wild1_file, sheet_name='bacteria_genus', skiprows=1)
            except:
                df_w1 = pd.read_excel(wild1_file, skiprows=1) # Fallback to first sheet

        col_map = {c: 'Genus' if 'genus' in str(c).lower() else 'Sample' if 'sample' in str(c).lower() else 'Abundance' if 'abundance' in str(c).lower() else c for c in df_w1.columns}
        df_w1 = df_w1.rename(columns=col_map)
        
        if 'Sample' in df_w1.columns and 'Genus' in df_w1.columns:
            df_w1 = df_w1[['Sample', 'Genus', 'Abundance']].dropna()
            unique_w1 = df_w1['Sample'].unique()
            mapping = {s: f"W_{w_count + i}" for i, s in enumerate(unique_w1)}
            w_count += len(unique_w1)
            
            df_w1['Sample'] = df_w1['Sample'].map(mapping)
            df_w1['Type'] = 'Wild_1'
            all_real_data.append(df_w1)
            print(f"   => Loaded {len(unique_w1)} Real Wild_1 Samples.")
        else:
            print(f"   [!] Error: Required columns missing. Found: {df_w1.columns.tolist()}")
    except Exception as e:
        print(f"   [!] Error reading {wild1_file}: {e}")

    # --- 3. WILD DATA 2 ---
    print(f"\n--- 3. LOADING WILD DATA 2 ({wild2_file}) ---")
    try:
        try:
            df_w2 = pd.read_csv(wild2_file)
        except:
            try:
                df_w2 = pd.read_excel(wild2_file, sheet_name='Genus top 10')
            except:
                df_w2 = pd.read_excel(wild2_file) # Fallback to first sheet
                
        gen_col = [c for c in df_w2.columns if 'genus' in str(c).lower()]
        if gen_col:
            df_w2 = df_w2.rename(columns={gen_col[0]: 'Genus'})
            df_w2['Genus'] = df_w2['Genus'].astype(str).str.replace('g__', '').replace('Others', '< 1.0%')
            
            target_cols = ['FZ', 'FJ', 'FH', 'YM', 'YZ', 'YL', 'YW']
            found_cols = [c for c in target_cols if c in df_w2.columns]
            
            if found_cols:
                df_melted = df_w2.melt(id_vars=['Genus'], value_vars=found_cols, var_name='Old_Sample', value_name='Abundance')
                if df_melted['Abundance'].max() < 1.1:
                    df_melted['Abundance'] *= 100
                    
                unique_w2 = df_melted['Old_Sample'].unique()
                mapping = {s: f"W_{w_count + i}" for i, s in enumerate(unique_w2)}
                w_count += len(unique_w2)
                
                df_melted['Sample'] = df_melted['Old_Sample'].map(mapping)
                df_melted = df_melted.drop(columns=['Old_Sample'])
                df_melted['Type'] = 'Wild_2'
                all_real_data.append(df_melted)
                print(f"   => Loaded {len(unique_w2)} Real Wild_2 Samples.")
            else:
                print(f"   [!] Error: Could not find columns {target_cols}. Found: {df_w2.columns.tolist()}")
        else:
            print(f"   [!] Error: Could not find 'genus' column. Found: {df_w2.columns.tolist()}")
    except Exception as e:
        print(f"   [!] Error reading {wild2_file}: {e}")

    # --- 4. MERGE & PIVOT ---
    print("\n--- 4. MERGING REAL DATA ---")
    if not all_real_data:
        print("CRITICAL EXIT: No data successfully loaded.")
        return

    master_real_df = pd.concat(all_real_data, ignore_index=True)
    pivot_df = master_real_df.pivot_table(index=['Sample', 'Type'], columns='Genus', values='Abundance', fill_value=0).reset_index()
    
    # --- 5. SYNTHETIC AUGMENTATION ---
    print("\n--- 5. GENERATING BALANCED SYNTHETICS (Total: 300) ---")
    
    amdb_parents = pivot_df[pivot_df['Type'] == 'AMDB']
    wild_parents = pivot_df[pivot_df['Type'].isin(['Wild_1', 'Wild_2'])]
    
    num_amdb_real = len(amdb_parents)
    num_wild_real = len(wild_parents)
    
    if num_wild_real == 0 or num_amdb_real == 0:
        print(f"CRITICAL EXIT: Missing Baseline Data. (AMDB: {num_amdb_real}, Wild: {num_wild_real})")
        return
    
    amdb_syn_needed = max(0, target_per_group - num_amdb_real)
    wild_syn_needed = max(0, target_per_group - num_wild_real)
    
    print(f"   -> AMDB: {num_amdb_real} Real. Generating {amdb_syn_needed} Synthetics.")
    print(f"   -> WILD: {num_wild_real} Real. Generating {wild_syn_needed} Synthetics.")
    
    feature_cols = [c for c in pivot_df.columns if c not in ['Sample', 'Type']]
    synthetic_rows = []
    
    for i in range(amdb_syn_needed):
        parent = amdb_parents.sample(1).iloc[0]
        vals = parent[feature_cols].values.astype(float)
        noise = np.random.normal(0, 0.01, size=vals.shape)
        new_vals = np.clip(vals + noise, 0, None)
        if new_vals.sum() > 0: new_vals = (new_vals / new_vals.sum()) * vals.sum()
        synthetic_rows.append({'Sample': f"synt_{parent['Sample']}_v{i+1}", 'Type': 'Syn_AMDB', **dict(zip(feature_cols, new_vals))})
        
    for i in range(wild_syn_needed):
        parent = wild_parents.sample(1).iloc[0]
        vals = parent[feature_cols].values.astype(float)
        noise = np.random.normal(0, 0.01, size=vals.shape)
        new_vals = np.clip(vals + noise, 0, None)
        if new_vals.sum() > 0: new_vals = (new_vals / new_vals.sum()) * vals.sum()
        synthetic_rows.append({'Sample': f"synt_{parent['Sample']}_v{i+1}", 'Type': f"Syn_{parent['Type']}", **dict(zip(feature_cols, new_vals))})

    final_dataset = pd.concat([pivot_df, pd.DataFrame(synthetic_rows)], ignore_index=True)
    final_dataset.to_csv(output_filename, index=False)
    
    print("\n" + "="*45)
    print(f"DONE! Dataset saved to: {output_filename}")
    print("Dataset Snapshot:")
    print(final_dataset['Type'].value_counts())
    print("="*45)

if __name__ == "__main__":
    build_balanced_master_dataset_final()