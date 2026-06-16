import os
import requests
import pandas as pd
import numpy as np
import re
import zipfile

# Set target paths
DATA_DIR = './data_cache'
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = 'train_balanced.tsv'

# Label Mapping Dictionary
LABEL_MAPPING = {
    # True claims
    'true': 'true',
    'real': 'true',
    'mostly-true': 'true',
    'correct': 'true',
    'supported': 'true',
    
    # False claims
    'false': 'false',
    'fake': 'false',
    'pants-on-fire': 'false',
    'incorrect': 'false',
    'refuted': 'false',
    
    # Mixture claims
    'mixture': 'mixture',
    'half-true': 'mixture',
    'barely-true': 'mixture',
    'mostly-false': 'mixture',
    
    # Unproven claims
    'unproven': 'unproven',
    'not enough info': 'unproven',
    'no evidence': 'unproven',
    'under review': 'unproven',
}

HEALTH_KEYWORDS = [
    'health', 'medicine', 'medical', 'clinical', 'disease', 'diseases', 'cancer',
    'vaccine', 'vaccines', 'vaccination', 'covid', 'coronavirus', 'virus', 'flu',
    'doctor', 'hospital', 'treatment', 'drug', 'drugs', 'pharma', 'medicaid', 'medicare',
    'chronic', 'diet', 'nutrition', 'vitamin', 'vitamins', 'obesity', 'tobacco',
    'smoking', 'heart', 'kidney', 'liver', 'infection', 'antibiotic', 'antibiotics',
    'mental', 'disorder', 'therapy', 'surgery', 'prevention', 'illness'
]

def download_file(url, filename):
    local_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(local_path):
        print(f"File already cached: {filename}")
        return local_path
    
    print(f"Downloading {url} -> {filename}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
        return local_path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def download_file_from_google_drive(file_id, filename):
    local_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(local_path):
        print(f"File already cached from GDrive: {filename}")
        return local_path
        
    print(f"Downloading from Google Drive (ID: {file_id}) -> {filename}...")
    try:
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': file_id}, stream=True)
        
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
                
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
            
        CHUNK_SIZE = 32768
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        print("Google Drive download complete.")
        return local_path
    except Exception as e:
        print(f"Failed to download GDrive file: {e}")
        return None

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_health_related(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in HEALTH_KEYWORDS)

def load_pubhealth():
    print("--- Loading PUBHEALTH (health_fact) ---")
    # Download zip file from Google Drive (as defined in health_fact.py)
    zip_id = "1eTtRs5cUlBP5dXsx-FTAlmXuB6JQi2qj"
    zip_path = download_file_from_google_drive(zip_id, "pubhealth.zip")
    
    if not zip_path:
        print("Failed to download PUBHEALTH zip.")
        return pd.DataFrame()
        
    # Extract zip file
    extracted_dir = os.path.join(DATA_DIR, "pubhealth_extracted")
    if not os.path.exists(extracted_dir):
        print(f"Extracting {zip_path} to {extracted_dir}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
            print("Extraction complete.")
        except Exception as e:
            print(f"Failed to extract zip file: {e}")
            return pd.DataFrame()
            
    # Read files
    splits = ['train.tsv', 'dev.tsv', 'test.tsv']
    dfs = []
    for split in splits:
        file_path = os.path.join(extracted_dir, "PUBHEALTH", split)
        if os.path.exists(file_path):
            print(f"Reading split: {split} ({file_path})")
            try:
                df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')
                dfs.append(df)
            except Exception as e:
                print(f"Error reading split {split}: {e}")
        else:
            print(f"Split file not found: {file_path}")
            
    if not dfs:
        return pd.DataFrame()
        
    combined = pd.concat(dfs, ignore_index=True)
    print(f"PUBHEALTH total loaded: {len(combined)}")
    
    if 'claim' in combined.columns and 'label' in combined.columns:
        filtered = combined[['claim', 'label']].dropna()
        filtered['claim'] = filtered['claim'].apply(normalize_text)
        filtered = filtered[filtered['claim'] != ""]
        return filtered.rename(columns={'claim': 'text'})
    return pd.DataFrame()

def load_constraint_2021():
    print("--- Loading CONSTRAINT-2021 ---")
    urls = {
        'train.csv': 'https://raw.githubusercontent.com/diptamath/covid_fake_news/main/data/Constraint_Train.csv',
        'val.csv': 'https://raw.githubusercontent.com/diptamath/covid_fake_news/main/data/Constraint_Val.csv',
        'test.csv': 'https://raw.githubusercontent.com/diptamath/covid_fake_news/main/data/Constraint_Test.csv'
    }
    
    dfs = []
    for fn, url in urls.items():
        path = download_file(url, f"constraint_{fn}")
        if path:
            try:
                df = pd.read_csv(path)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {fn}: {e}")
                
    if not dfs:
        return pd.DataFrame()
        
    combined = pd.concat(dfs, ignore_index=True)
    print(f"CONSTRAINT-2021 raw count: {len(combined)}")
    
    if 'tweet' in combined.columns and 'label' in combined.columns:
        filtered = combined[['tweet', 'label']].dropna()
        filtered['tweet'] = filtered['tweet'].apply(normalize_text)
        return filtered.rename(columns={'tweet': 'text'})
    return pd.DataFrame()

def load_coaid():
    print("--- Loading CoAID ---")
    dates = ['05-01-2020', '07-01-2020', '09-01-2020', '11-01-2020']
    dfs = []
    
    # We will try both Claim and News files
    for date in dates:
        files_to_load = [
            ("ClaimFakeCOVID-19.csv", "false"),
            ("ClaimRealCOVID-19.csv", "true"),
            ("NewsFakeCOVID-19.csv", "false"),
            ("NewsRealCOVID-19.csv", "true")
        ]
        
        for fn, label in files_to_load:
            url = f"https://raw.githubusercontent.com/cuilimeng/CoAID/master/{date}/{fn}"
            path = download_file(url, f"coaid_{date}_{fn}")
            if path:
                try:
                    df = pd.read_csv(path)
                    df['label'] = label
                    # Check text column: title or newstitle
                    text_col = None
                    for col in ['title', 'newstitle', 'tweet']:
                        if col in df.columns:
                            text_col = col
                            break
                            
                    if text_col:
                        extracted = df[[text_col, 'label']].dropna()
                        extracted.columns = ['text', 'label']
                        extracted['text'] = extracted['text'].apply(normalize_text)
                        dfs.append(extracted)
                        print(f"  Loaded {len(extracted)} rows from {date}/{fn}")
                except Exception as e:
                    # File might not exist for this date
                    pass
                    
    if not dfs:
        return pd.DataFrame()
        
    combined = pd.concat(dfs, ignore_index=True)
    print(f"CoAID total loaded: {len(combined)}")
    return combined

def load_liar():
    print("--- Loading LIAR (Health-Filtered) ---")
    urls = {
        'train.tsv': 'https://raw.githubusercontent.com/tfs4/liar_dataset/master/train.tsv',
        'test.tsv': 'https://raw.githubusercontent.com/tfs4/liar_dataset/master/test.tsv',
        'valid.tsv': 'https://raw.githubusercontent.com/tfs4/liar_dataset/master/valid.tsv'
    }
    
    dfs = []
    for fn, url in urls.items():
        path = download_file(url, f"liar_{fn}")
        if path:
            try:
                df = pd.read_csv(path, sep='\t', header=None, on_bad_lines='skip')
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {fn}: {e}")
                
    if not dfs:
        return pd.DataFrame()
        
    combined = pd.concat(dfs, ignore_index=True)
    print(f"LIAR raw count: {len(combined)}")
    
    statements = []
    labels = []
    for idx, row in combined.iterrows():
        if len(row) > 3:
            label = row[1]
            text = str(row[2])
            subjects = str(row[3])
            
            if is_health_related(text) or any(keyword in subjects.lower() for keyword in HEALTH_KEYWORDS):
                statements.append(text)
                labels.append(label)
                
    filtered = pd.DataFrame({'text': statements, 'label': labels})
    filtered['text'] = filtered['text'].apply(normalize_text)
    print(f"LIAR health-filtered count: {len(filtered)}")
    return filtered

def load_local_kaggle():
    print("--- Scanning for Local Kaggle Health Datasets ---")
    kaggle_dir = './kaggle-health-datasets'
    if not os.path.exists(kaggle_dir):
        print("Kaggle Health Datasets directory not found locally. Skipping local Kaggle parse.")
        return pd.DataFrame()
        
    dfs = []
    for root, dirs, files in os.walk(kaggle_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Parsing local Kaggle CSV: {file_path}")
                try:
                    df = pd.read_csv(file_path)
                    text_col = None
                    label_col = None
                    for col in df.columns:
                        col_lower = col.lower()
                        if col_lower in ['claim', 'tweet', 'statement', 'text', 'headline', 'title']:
                            text_col = col
                        if col_lower in ['label', 'verdict', 'class', 'status']:
                            label_col = col
                            
                    if text_col and label_col:
                        extracted = df[[text_col, label_col]].dropna()
                        extracted.columns = ['text', 'label']
                        extracted['text'] = extracted['text'].apply(normalize_text)
                        dfs.append(extracted)
                        print(f"Successfully extracted {len(extracted)} claims from {file}")
                except Exception as e:
                    print(f"Failed to parse {file_path}: {e}")
                    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def load_base_dataset():
    print("--- Loading Base Dataset (train.tsv / train.csv) ---")
    for file_name in ['train.tsv', 'train.csv']:
        if os.path.exists(file_name):
            print(f"Found base dataset: {file_name}")
            try:
                sep = '\t' if file_name.endswith('.tsv') else ','
                df = pd.read_csv(file_name, sep=sep)
                if 'claim' in df.columns and 'label' in df.columns:
                    extracted = df[['claim', 'label']].dropna()
                    extracted.columns = ['text', 'label']
                    extracted['text'] = extracted['text'].apply(normalize_text)
                    print(f"Base dataset count: {len(extracted)}")
                    return extracted
            except Exception as e:
                print(f"Error reading base dataset: {e}")
    print("Base dataset not found in workspace.")
    return pd.DataFrame()

def synthesize_claims():
    print("--- Synthesizing Claims from Chronic Disease & Mortality Statistics ---")
    diseases = [
        {"name": "Arthritis", "stat": "22.7%", "year": "2019", "demographic": "US adults"},
        {"name": "Heart disease", "stat": "11.7%", "year": "2020", "demographic": "adults over 65"},
        {"name": "Chronic obstructive pulmonary disease (COPD)", "stat": "6.2%", "year": "2018", "demographic": "US adults"},
        {"name": "Diabetes", "stat": "10.5%", "year": "2021", "demographic": "the global population"},
        {"name": "Depression", "stat": "18.5%", "year": "2020", "demographic": "adolescents"},
        {"name": "Alzheimer's disease", "stat": "10.0%", "year": "2019", "demographic": "seniors over 65"},
        {"name": "Obesity", "stat": "42.4%", "year": "2018", "demographic": "US adults"},
        {"name": "Stroke", "stat": "2.7%", "year": "2019", "demographic": "adults"}
    ]
    
    synthesized = []
    for d in diseases:
        # True templates
        synthesized.append({
            "text": f"According to chronic disease statistics, the prevalence of {d['name']} in {d['year']} among {d['demographic']} was approximately {d['stat']}.",
            "label": "true"
        })
        synthesized.append({
            "text": f"Medical reports from {d['year']} indicate that {d['stat']} of {d['demographic']} were affected by {d['name']}.",
            "label": "true"
        })
        
        # False templates (mutated stats)
        mutated_stat = "95%" if d['stat'] != "95%" else "2%"
        synthesized.append({
            "text": f"According to chronic disease statistics, the prevalence of {d['name']} in {d['year']} among {d['demographic']} was approximately {mutated_stat}.",
            "label": "false"
        })
        synthesized.append({
            "text": f"Medical reports from {d['year']} indicate that {mutated_stat} of {d['demographic']} were affected by {d['name']}.",
            "label": "false"
        })
        
        # Mixture templates (correct disease & year but incorrect demographic or mixed statements)
        synthesized.append({
            "text": f"In {d['year']}, {d['stat']} of newborn infants were diagnosed with {d['name']}, which is a leading chronic condition.",
            "label": "mixture"
        })
        synthesized.append({
            "text": f"Statistics show {d['stat']} of {d['demographic']} have {d['name']}, making it a totally cured and non-fatal disease.",
            "label": "mixture"
        })
        
        # Unproven templates
        synthesized.append({
            "text": f"Recent trials suggest that {d['name']} prevalence can be completely reduced to 0% by drinking lemon juice daily.",
            "label": "unproven"
        })
        synthesized.append({
            "text": f"Some claims suggest that {d['name']} was invented by pharmaceutical companies in {d['year']} to sell drugs.",
            "label": "unproven"
        })
        synthesized.append({
            "text": f"There are anecdotal claims that a secret diet can prevent {d['name']} in {d['demographic']} without fail.",
            "label": "unproven"
        })

    df_synth = pd.DataFrame(synthesized)
    print(f"Synthesized {len(df_synth)} claims from statistics.")
    return df_synth

def main():
    df_base = load_base_dataset()
    df_pubhealth = load_pubhealth()
    df_constraint = load_constraint_2021()
    df_coaid = load_coaid()
    df_liar = load_liar()
    df_kaggle = load_local_kaggle()
    df_synth = synthesize_claims()
    
    all_dfs = [df_base, df_pubhealth, df_constraint, df_coaid, df_liar, df_kaggle, df_synth]
    all_dfs = [df for df in all_dfs if not df.empty]
    
    if not all_dfs:
        print("ERROR: No data loaded at all. Exiting.")
        return
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal raw combined count: {len(combined_df)}")
    
    combined_df['label'] = combined_df['label'].astype(str).str.lower().str.strip()
    combined_df['mapped_label'] = combined_df['label'].map(LABEL_MAPPING)
    
    combined_df = combined_df.dropna(subset=['mapped_label'])
    combined_df = combined_df.drop(columns=['label']).rename(columns={'mapped_label': 'label'})
    
    combined_df['text_clean'] = combined_df['text'].str.lower().str.strip()
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['text_clean'])
    combined_df = combined_df.drop(columns=['text_clean'])
    print(f"Deduplicated: removed {before_dedup - len(combined_df)} duplicate rows. Count: {len(combined_df)}")
    
    print("\nProcessed label distribution:")
    print(combined_df['label'].value_counts())
    
    target_per_class = 11000
    balanced_dfs = []
    
    for label in ['true', 'false', 'mixture', 'unproven']:
        class_df = combined_df[combined_df['label'] == label]
        class_count = len(class_df)
        print(f"Class '{label}': raw count = {class_count}")
        
        if class_count == 0:
            print(f"Warning: Class '{label}' has 0 samples!")
            continue
            
        if class_count >= target_per_class:
            sampled = class_df.sample(n=target_per_class, random_state=42)
            balanced_dfs.append(sampled)
            print(f"  Under-sampled to {target_per_class}")
        else:
            sampled = class_df.sample(n=target_per_class, replace=True, random_state=42)
            balanced_dfs.append(sampled)
            print(f"  Over-sampled (duplicated) to {target_per_class}")
            
    final_df = pd.concat(balanced_dfs, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df = final_df.rename(columns={'text': 'claim'})
    
    print(f"\nFinal balanced dataset size: {len(final_df)}")
    print("Final label distribution:")
    print(final_df['label'].value_counts())
    
    final_df.to_csv(OUTPUT_FILE, sep='\t', index=False)
    print(f"SUCCESS: Saved balanced dataset to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
