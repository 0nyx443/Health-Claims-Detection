import os
import pandas as pd
import requests

def main():
    print("1. Loading original train.tsv...")
    if not os.path.exists('train.tsv'):
        print("Error: train.tsv not found in the root directory.")
        return
        
    df_orig = pd.read_csv('train.tsv', sep='\t')
    print(f"Original dataset size: {len(df_orig)}")
    
    # Backup original
    if not os.path.exists('train_original.tsv'):
        df_orig.to_csv('train_original.tsv', sep='\t', index=False)
        print("Saved backup of original dataset as 'train_original.tsv'.")
    
    print("2. Downloading HealthFC dataset from GitHub...")
    url = "https://raw.githubusercontent.com/jvladika/HealthFC/main/Datensatz.csv"
    try:
        df_fc = pd.read_csv(url)
        print(f"Downloaded HealthFC dataset. Loaded {len(df_fc)} samples.")
    except Exception as e:
        print(f"Failed to download HealthFC dataset: {e}")
        return
        
    print("3. Transforming HealthFC columns to match project format...")
    # Map HealthFC label: 0 -> true, 1 -> unproven, 2 -> false
    label_map_fc = {0: 'true', 1: 'unproven', 2: 'false'}
    
    fc_rows = []
    for idx, row in df_fc.iterrows():
        mapped_label = label_map_fc.get(row['label'], 'unproven')
        fc_rows.append({
            'claim_id': f"healthfc_{idx}",
            'claim': row['en_claim'],
            'date_published': row['date'] if pd.notna(row['date']) else '',
            'explanation': row['en_explanation'] if pd.notna(row['en_explanation']) else '',
            'fact_checkers': 'HealthFC Medical Experts',
            'main_text': row['en_top_sentences'] if pd.notna(row['en_top_sentences']) else '',
            'sources': row['url'] if pd.notna(row['url']) else '',
            'label': mapped_label,
            'subjects': 'medicine'
        })
    df_fc_transformed = pd.DataFrame(fc_rows)
    
    print("4. Adding common medical myths and facts for data priming...")
    # We prime the dataset with clear, short consumer health facts and myths
    priming_data = [
        # True claims
        ("Eating fruits and vegetables provides important vitamins and minerals.", "true"),
        ("Smoking increases the risk of cancer and lung disease.", "true"),
        ("Washing your hands helps prevent the spread of infections.", "true"),
        ("Too much sugar intake can increase the risk of obesity and tooth decay.", "true"),
        ("Wearing sunscreen helps reduce the risk of skin cancer.", "true"),
        ("Exercising regularly strengthens the cardiovascular system.", "true"),
        ("Drinking enough water is essential for kidney function.", "true"),
        ("Antibiotics cure bacterial infections but do not work on viruses.", "true"),
        ("High blood pressure increases the risk of heart disease and stroke.", "true"),
        ("A balanced diet supports a healthy immune system.", "true"),
        
        # False claims
        ("Drinking lemon water cures cancer.", "false"),
        ("Vaccines cause autism.", "false"),
        ("Detox teas remove toxins from your body.", "false"),
        ("Microwave ovens make food radioactive.", "false"),
        ("Apple cider vinegar melts belly fat without diet or exercise.", "false"),
        ("Eating fat makes you fat instantly.", "false"),
        ("Organic food is always 100% pesticide-free.", "false"),
        ("Shaving makes hair grow back thicker and faster.", "false"),
        ("Cracking your knuckles causes arthritis.", "false"),
        ("Cold weather causes the common cold.", "false")
    ]
    
    priming_rows = []
    for idx, (claim, label) in enumerate(priming_data):
        priming_rows.append({
            'claim_id': f"priming_{idx}",
            'claim': claim,
            'date_published': '2026-06-05',
            'explanation': 'Fact-checked health assertion.',
            'fact_checkers': 'Medical consensus',
            'main_text': claim,
            'sources': 'WHO/CDC',
            'label': label,
            'subjects': 'medicine'
        })
    df_priming = pd.DataFrame(priming_rows)
    
    print("5. Merging datasets...")
    df_merged = pd.concat([df_orig, df_fc_transformed, df_priming], ignore_index=True)
    print(f"Merged dataset size: {len(df_merged)}")
    print("New class distribution:")
    print(df_merged['label'].value_counts())
    
    print("6. Overwriting train.tsv with the merged dataset...")
    df_merged.to_csv('train.tsv', sep='\t', index=False)
    print("SUCCESS: train.tsv has been updated with HealthFC and priming data.")

if __name__ == '__main__':
    main()
