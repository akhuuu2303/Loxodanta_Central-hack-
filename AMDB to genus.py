import pandas as pd
import os

def convert_amdb_to_vae_format(input_folder, output_folder):
    """
    Converts AMDB files (Sample N1.csv, etc.) to a Genus-level summary
    Removing 'Sex', 'Age', and 'Maturity' to keep data lean.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_processed_dfs = []

    # 1. Loop through your natural samples folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv") and filename.startswith("Sample"):
            file_path = os.path.join(input_folder, filename)
            
            # Load the individual elephant file
            df = pd.read_csv(file_path)
            sample_id = filename.replace(".csv", "")
            
            # 2. Split the 'Taxonomy' string into biological levels
            def split_taxonomy(tax_str):
                if pd.isna(tax_str): return ['Unassigned']*6
                parts = str(tax_str).split(';')
                while len(parts) < 6:
                    parts.append('Unassigned')
                return parts[:6]

            tax_data = df['Taxonomy'].apply(split_taxonomy).tolist()
            tax_df = pd.DataFrame(tax_data, columns=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus'])
            
            # 3. Aggregate abundance by Genus
            abundance_col = 'Relative abundance (%)'
            # (Safety check if column name is slightly different)
            if abundance_col not in df.columns:
                possible = [c for c in df.columns if 'abundance' in c.lower()]
                if possible: abundance_col = possible[0]

            df_combined = pd.concat([tax_df, df[abundance_col]], axis=1)
            genus_summary = df_combined.groupby(['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus'])[abundance_col].sum().reset_index()
            
            # 4. Add Sample and Name only
            genus_summary['Sample'] = sample_id
            genus_summary['Name'] = sample_id
            
            # 5. Standardize column name
            genus_summary = genus_summary.rename(columns={abundance_col: 'Relative Abundance (%)'})
            
            # 6. Reorder and save individual file
            final_cols = ['Sample', 'Name', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Relative Abundance (%)']
            genus_summary = genus_summary[final_cols]
            
            genus_summary.to_csv(os.path.join(output_folder, f"genus_{filename}"), index=False)
            all_processed_dfs.append(genus_summary)
            print(f"Cleaned and Processed: {filename}")

    # 7. CREATE THE MASTER TRAINING MATRIX
    if all_processed_dfs:
        full_df = pd.concat(all_processed_dfs, ignore_index=True)
        # Pivot so Rows = Elephants and Columns = Genus (The AI features)
        master_matrix = full_df.pivot_table(
            index=['Sample', 'Name'],
            columns='Genus',
            values='Relative Abundance (%)',
            fill_value=0
        ).reset_index()
        
        # Save the finalized training data
        master_matrix.to_csv('master_vae_training_data.csv', index=False)
        print("\nSUCCESS: 'master_vae_training_data.csv' is ready for VAE training!")
        return master_matrix

# --- RUN THE CLEANED CONVERSION ---
# Make sure your files are in 'natural_samples' folder
convert_amdb_to_vae_format(input_folder='natural_samples', output_folder='processed_amdb_genus')