import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DISTILBERT_DIR = './distilbert_health_model'

test_claims = [
    # True claims
    "Smoking increases the risk of cancer and lung disease.",
    "Washing your hands helps prevent the spread of infections.",
    "Vaccination is a proven method to prevent infectious diseases like measles.",
    
    # False claims
    "Drinking lemon water cures cancer completely.",
    "Drinking bleach cures covid19.",
    "Vaccines cause autism.",
    
    # Mixture claims
    "Drinking 8 glasses of water per day is necessary for everyone to maintain proper hydration.",
    "Arthritis is a chronic disease that can be completely cured by drinking warm water before eating.",
    
    # Unproven / Out-of-Distribution claims
    "Eating purple mushrooms makes humans fly.",
    "A secret herbal tea cures all types of mental disorders without side effects.",
]

def clean_and_normalize_text(text):
    import re
    text = str(text)
    text = re.sub(r'[\U00010000-\U0010FFFF]', '', text)
    text = re.sub(r'[\u2600-\u27BF]', '', text)
    text = text.lower().strip()
    abbreviations = {r'\bb4\b': 'before', r'\bw/\b': 'with', r'\bw/o\b': 'without', r'\b&\b': 'and', r'\b2\b': 'too'}
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)
    spelling_map = {"vacines": "vaccines", "vacine": "vaccine", "drnking": "drinking", "cancr": "cancer"}
    words = text.split()
    return " ".join([spelling_map.get(w, w) for w in words])

def test_model():
    print(f"Loading DistilBERT from: {DISTILBERT_DIR}")
    if not os.path.exists(DISTILBERT_DIR):
        print("Error: distilbert_health_model directory not found!")
        return

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
    model.eval()

    labels = ['true', 'false', 'mixture', 'unproven']
    print("\n--- Running Predictions ---")
    for claim in test_claims:
        normalized = clean_and_normalize_text(claim)
        inputs = tokenizer(normalized, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        max_prob, pred_idx = torch.max(probs, dim=1)
        
        confidence = max_prob.item()
        prediction = labels[pred_idx.item()]
        
        # Determine safety verdict using 70% threshold
        if confidence < 0.70:
            verdict = "unproven (low confidence override)"
        else:
            verdict = prediction
            
        print(f"Claim: '{claim}'")
        print(f"  -> Prediction: {prediction.upper()} (Confidence: {confidence:.2%})")
        print(f"  -> Final Verdict: {verdict.upper()}")
        print("-" * 50)

if __name__ == '__main__':
    test_model()
