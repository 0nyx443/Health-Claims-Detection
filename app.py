import streamlit as st
import nltk
import joblib
import string
import os
import re
import difflib
import html
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    TRANSFORMERS_AVAILABLE = False

# Set page config without emojis
st.set_page_config(page_title="Health Fact Checker", page_icon=None, layout="centered")

# Download resources silently
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Define directories/filenames
DISTILBERT_DIR = './distilbert_health_model'
SVM_MODEL_PATH = 'health_claim_model.pkl'
SVM_VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
DISTILBERT_CONFIDENCE_THRESHOLD = 0.70
FUZZY_KB_CONFIDENCE_THRESHOLD = 0.85
RISKY_WORDS = [
    "cure",
    "cures",
    "guaranteed",
    "miracle",
    "100%",
    "always",
    "never",
    "instant",
    "permanent",
    "no side effects",
]

# Preprocessing utilities
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_and_normalize_text(text):
    text = str(text)
    # Strip emojis
    text = re.sub(r'[\U00010000-\U0010FFFF]', '', text)
    text = re.sub(r'[\u2600-\u27BF]', '', text)
    text = text.lower().strip()
    
    # Expand abbreviations
    abbreviations = {
        r'\bb4\b': 'before',
        r'\bw/\b': 'with',
        r'\bw/o\b': 'without',
        r'\b&\b': 'and',
        r'\b2\b': 'too'
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)
        
    # Standardize spelling of common terms to prevent OOV issues
    spelling_map = {
        "vacines": "vaccines",
        "vacine": "vaccine",
        "drnking": "drinking",
        "cancr": "cancer",
        "smokng": "smoking",
        "desease": "disease",
        "infetcion": "infection",
        "healty": "healthy",
        "vitamns": "vitamins"
    }
    words = text.split()
    corrected_words = [spelling_map.get(w, w) for w in words]
    return " ".join(corrected_words)

TAGLOG_ENGLISH_MAP = {
    "nagdudulot ng": "causes",
    "nagdudulot": "causes",
    "nagko-cause": "causes",
    "nagcocause": "causes",
    "nag-cause": "caused",
    "umiwas sa": "avoid",
    "nakakaiwas sa": "prevents",
    "nakakaiwas": "prevents",
    "maiwasan": "avoid",
    "gumaling sa": "cure",
    "para gumaling sa": "to cure",
    "para gumaling": "to cure",
    "gumaling": "heal",
    "gamot sa": "cure for",
    "gamot": "medicine",
    "sakit sa kidney": "kidney disease",
    "sakit sa puso": "heart disease",
    "sakit": "disease",
    "bata": "child",
    "mga bata": "children",
    "bago kumain": "before eating",
    "bago": "before",
    "kumain": "eat",
    "uminom ng": "drink",
    "uminom": "drink",
    "maraming": "many",
    "marami": "much",
    "masama sa": "bad for",
    "masama": "bad",
    "mabuti sa": "good for",
    "mabuti": "good",
    "mainit na": "hot",
    "mainit": "hot",
    "tubig": "water",
    "gulay": "vegetables",
    "prutas": "fruits",
    "kumain ng": "eat",
    "kumain": "eat",
    "ang": "the",
    "ay": "is",
    "ng": "of",
    "sa": "in",
    "mga": "",
    "para": "for",
}

def translate_taglish_to_english(text):
    text = text.lower()
    sorted_keys = sorted(TAGLOG_ENGLISH_MAP.keys(), key=len, reverse=True)
    for key in sorted_keys:
        val = TAGLOG_ENGLISH_MAP[key]
        escaped_key = re.escape(key)
        if key.replace(" ", "").isalnum():
            pattern = rf'\b{escaped_key}\b'
        else:
            pattern = escaped_key
        text = re.sub(pattern, val, text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_text_svm(text):
    normalized = clean_and_normalize_text(text)
    translated = translate_taglish_to_english(normalized)
    
    cleaned = translated.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(cleaned)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

def check_static_kb(text):
    cleaned = clean_and_normalize_text(text)
    # Strip standard punctuation for lookup comparison
    cleaned_no_punct = cleaned.translate(str.maketrans('', '', string.punctuation)).strip()
    translated = translate_taglish_to_english(cleaned_no_punct).strip()
    
    # 1. Exact match on normalized or translated
    if cleaned_no_punct in KNOWN_CLAIMS:
        return {
            "label": KNOWN_CLAIMS[cleaned_no_punct],
            "matched_claim": cleaned_no_punct,
            "match_type": "exact",
            "similarity": 1.0,
        }
    if translated in KNOWN_CLAIMS:
        return {
            "label": KNOWN_CLAIMS[translated],
            "matched_claim": translated,
            "match_type": "exact",
            "similarity": 1.0,
        }
        
    # 2. Fuzzy match on normalized (cutoff 0.60)
    matches_cleaned = difflib.get_close_matches(cleaned_no_punct, KNOWN_CLAIMS.keys(), n=1, cutoff=0.60)
    if matches_cleaned:
        similarity = difflib.SequenceMatcher(None, cleaned_no_punct, matches_cleaned[0]).ratio()
        return {
            "label": KNOWN_CLAIMS[matches_cleaned[0]],
            "matched_claim": matches_cleaned[0],
            "match_type": "fuzzy",
            "similarity": similarity,
        }
        
    # 3. Fuzzy match on translated (cutoff 0.60)
    matches_translated = difflib.get_close_matches(translated, KNOWN_CLAIMS.keys(), n=1, cutoff=0.60)
    if matches_translated:
        similarity = difflib.SequenceMatcher(None, translated, matches_translated[0]).ratio()
        return {
            "label": KNOWN_CLAIMS[matches_translated[0]],
            "matched_claim": matches_translated[0],
            "match_type": "fuzzy",
            "similarity": similarity,
        }
        
    return None

def detect_risky_words(text):
    normalized = clean_and_normalize_text(text)
    detected = []
    for term in RISKY_WORDS:
        if " " in term or "%" in term:
            if term in normalized:
                detected.append(term)
        elif re.search(rf"\b{re.escape(term)}\b", normalized):
            detected.append(term)
    return detected

def format_label(label):
    return str(label).strip().capitalize()

def determine_final_verdict(source, model_prediction, confidence=None, kb_match=None, risky_terms=None):
    risky_terms = risky_terms or []

    if source == "Knowledge Base":
        if kb_match and kb_match["match_type"] == "exact":
            return model_prediction, "This matched a known claim in the local knowledge base."

        similarity = kb_match["similarity"] if kb_match else 0
        if similarity < FUZZY_KB_CONFIDENCE_THRESHOLD:
            return "unproven", "A similar claim was found, but the match was not exact enough for a strong verdict."
        return model_prediction, "This closely matched a known claim in the local knowledge base, but it was not an exact match."

    if source == "DistilBERT":
        if confidence is None or confidence < DISTILBERT_CONFIDENCE_THRESHOLD:
            return "unproven", "The model confidence is low, so the system avoids making a strong health judgment."
        return model_prediction, "The model confidence is above the configured threshold."

    if source == "LinearSVC fallback":
        if risky_terms:
            return "unproven", "The claim uses strong medical wording and was checked using the fallback classifier, so the system avoids making a strong medical judgment."
        return model_prediction, "This is a fallback model prediction. Confidence is not available."

    return "unproven", "The system could not determine a reliable model source."

def get_base_model_source():
    if loaded_model["type"] == "distilbert":
        return "DistilBERT"
    return "LinearSVC fallback"

def get_confidence_text(source, confidence):
    if source == "DistilBERT" and confidence is not None:
        return f"{confidence:.0%}"
    if source == "LinearSVC fallback":
        return "Not available for LinearSVC fallback"
    if source == "Knowledge Base":
        return "Local knowledge-base match; not a model confidence score"
    return "Not available"


# Static Knowledge Base for high-frequency direct health facts and myths
KNOWN_CLAIMS = {
    # Medical Facts (True)
    "eating fruits and vegetables provides important vitamins and minerals": "true",
    "smoking increases the risk of cancer and lung disease": "true",
    "smoking causes lung disease": "true",
    "washing your hands helps prevent the spread of infections": "true",
    "washing hands helps prevent the spread of infections": "true",
    "washing hands prevents infection": "true",
    "washing hands before eating": "true",
    "wash hands before eating": "true",
    "too much sugar intake can increase the risk of obesity and tooth decay": "true",
    "wearing sunscreen helps reduce the risk of skin cancer": "true",
    "exercising regularly strengthens the cardiovascular system": "true",
    "drinking enough water is essential for kidney function": "true",
    "antibiotics cure bacterial infections but do not work on viruses": "true",
    "high blood pressure increases the risk of heart disease and stroke": "true",
    "a balanced diet supports a healthy immune system": "true",
    "sleeping on your left side prevents acid reflux": "true",
    
    # Medical Myths (False)
    "drinking lemon water cures cancer": "false",
    "drinking lemon juice cures cancer": "false",
    "vaccines cause autism": "false",
    "detox teas remove toxins from your body": "false",
    "microwave ovens make food radioactive": "false",
    "apple cider vinegar melts belly fat without diet or exercise": "false",
    "eating fat makes you fat instantly": "false",
    "organic food is always 100 pesticidefree": "false",
    "organic food is always 100 pesticide free": "false",
    "shaving makes hair grow back thicker and faster": "false",
    "cracking your knuckles causes arthritis": "false",
    "cold weather causes the common cold": "false",
    "drinking bleach cures covid19": "false",
    "drinking bleach cures covid 19": "false",
    "drinking bleach cures covid-19": "false",
    "cancer is caused by positive thoughts and can be cured by meditation": "false"
}

# Loader for models
@st.cache_resource
def load_fact_checker_model():
    # Attempt to load DistilBERT
    if os.path.exists(DISTILBERT_DIR):
        if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            try:
                tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
                model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
                # Put in evaluation mode
                model.eval()
                return {
                    'type': 'distilbert',
                    'model': model,
                    'tokenizer': tokenizer,
                    'note': None,
                }
            except Exception as e:
                st.warning(f"Failed to load DistilBERT from {DISTILBERT_DIR}. Error: {e}. Falling back to baseline SVM model.")
        else:
            missing_packages = []
            if not TRANSFORMERS_AVAILABLE:
                missing_packages.append("transformers")
            if not TORCH_AVAILABLE:
                missing_packages.append("torch")
            st.warning(f"DistilBERT folder was found, but {', '.join(missing_packages)} is unavailable. Falling back to LinearSVC.")
    
    # Load fallback SVM model
    if os.path.exists(SVM_MODEL_PATH) and os.path.exists(SVM_VECTORIZER_PATH):
        try:
            model = joblib.load(SVM_MODEL_PATH)
            vectorizer = joblib.load(SVM_VECTORIZER_PATH)
            return {
                'type': 'svm',
                'model': model,
                'vectorizer': vectorizer,
                'note': "DistilBERT model not found. Using LinearSVC fallback." if not os.path.exists(DISTILBERT_DIR) else "Using LinearSVC fallback."
            }
        except Exception as e:
            st.error(f"Failed to load SVM baseline. Error: {e}")
            st.stop()
    else:
        st.error("No classification model found! Please train the model using Google Colab or train_model.py first.")
        st.stop()

# Load the available model
loaded_model = load_fact_checker_model()

st.markdown("""
<style>
.stApp {
    background-color: #f7f7f7;
}

html, body, [class*="css"], .stMarkdown {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #111111;
}

.title-container {
    padding-top: 32px;
    margin-bottom: 24px;
}

.title-text {
    font-size: 32px;
    font-weight: 700;
    color: #111111;
    letter-spacing: 0;
    margin-bottom: 8px;
}

.subtitle-text {
    font-size: 16px;
    color: #555555;
    margin-bottom: 0;
}

.input-card, .result-card, .footer-note {
    border: 1px solid #d8d8d8;
    background-color: #ffffff;
    border-radius: 8px;
    padding: 20px;
    margin: 18px 0;
}

.muted {
    color: #666666;
    font-size: 14px;
    line-height: 1.5;
}

.result-label {
    color: #666666;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0;
    margin-bottom: 4px;
}

.result-value {
    color: #111111;
    font-size: 18px;
    font-weight: 650;
    margin-bottom: 16px;
}

.final-value {
    font-size: 28px;
    margin-bottom: 18px;
}

.reason-text {
    color: #222222;
    font-size: 15px;
    line-height: 1.6;
}

.stTextArea textarea {
    border: 1px solid #cfcfcf !important;
    background-color: #ffffff !important;
    border-radius: 8px !important;
    font-size: 15px !important;
    color: #111111 !important;
    box-shadow: none !important;
}

.stTextArea textarea:focus {
    border-color: #111111 !important;
}

.stButton button {
    background-color: #111111 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}

.stButton button:hover {
    background-color: #333333 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-container">
    <div class="title-text">Health Claim Fact-Checker</div>
    <div class="subtitle-text">Check health-related claims using a trained text classifier.</div>
</div>
""", unsafe_allow_html=True)

user_input = st.text_area("Enter health claim", placeholder="e.g., Vitamin C cures colds")

preview_kb_match = check_static_kb(user_input) if user_input else None
preview_source = "Knowledge Base" if preview_kb_match else get_base_model_source()
st.markdown(
    f'<div class="muted">Current source: {html.escape(preview_source)}</div>',
    unsafe_allow_html=True,
)

if loaded_model.get("note"):
    with st.expander("Model status"):
        st.write(loaded_model["note"])

if st.button("Check Claim", type="primary"):
    if user_input:
        with st.spinner('Analyzing claim...'):
            model_prediction = None
            final_verdict = "unproven"
            reason = "The system could not determine a reliable verdict."
            confidence_score = None
            kb_match = check_static_kb(user_input)
            source_used = "Knowledge Base" if kb_match else get_base_model_source()
            normalized_input = clean_and_normalize_text(user_input)
            translated_input = translate_taglish_to_english(normalized_input)
            risky_terms = detect_risky_words(user_input)

            if kb_match is not None:
                model_prediction = kb_match["label"]
            else:
                # Step B: Pass to ML/DL models
                if loaded_model['type'] == 'distilbert':
                    tokenizer = loaded_model['tokenizer']
                    model = loaded_model['model']
                    
                    # Tokenize input claim
                    inputs = tokenizer(translated_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
                    
                    # Forward pass without calculating gradients
                    with torch.no_grad():
                        outputs = model(**inputs)
                        
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                    max_prob, pred_idx = torch.max(probs, dim=1)
                    confidence_score = max_prob.item()
                    
                    # Map back to labels
                    labels = ['true', 'false', 'mixture', 'unproven']
                    model_prediction = labels[pred_idx.item()]
                else:
                    # Baseline SVM prediction
                    svc_model = loaded_model['model']
                    tfidf = loaded_model['vectorizer']
                    
                    cleaned_input = preprocess_text_svm(user_input)
                    vectorized_input = tfidf.transform([cleaned_input])
                    model_prediction = svc_model.predict(vectorized_input)[0]

            final_verdict, reason = determine_final_verdict(
                source_used,
                model_prediction,
                confidence=confidence_score,
                kb_match=kb_match,
                risky_terms=risky_terms,
            )

            confidence_text = get_confidence_text(source_used, confidence_score)
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Final verdict</div>
                    <div class="result-value final-value">{html.escape(format_label(final_verdict))}</div>
                    <div class="result-label">Model prediction</div>
                    <div class="result-value">{html.escape(format_label(model_prediction))}</div>
                    <div class="result-label">Confidence</div>
                    <div class="result-value">{html.escape(confidence_text)}</div>
                    <div class="result-label">Source</div>
                    <div class="result-value">{html.escape(source_used)}</div>
                    <div class="result-label">Reason</div>
                    <div class="reason-text">{html.escape(reason)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Details"):
                st.write(f"Normalized input: {normalized_input}")
                st.write(f"Translated input: {translated_input}")
                st.write(f"Source used: {source_used}")
                st.write(f"Raw model prediction: {model_prediction}")
                st.write(f"Final verdict: {final_verdict}")
                st.write(f"Active model folder status: {'found' if os.path.exists(DISTILBERT_DIR) else 'not found'}")
                if confidence_score is not None:
                    st.write(f"Confidence score: {confidence_score:.4f}")
                else:
                    st.write("Confidence score: not available")
                if risky_terms:
                    st.write(f"Risky wording detected: {', '.join(risky_terms)}")
                else:
                    st.write("Risky wording detected: none")
                if kb_match is not None:
                    st.write(f"Knowledge-base match type: {kb_match['match_type']}")
                    st.write(f"Matched claim: {kb_match['matched_claim']}")
                    st.write(f"Similarity score: {kb_match['similarity']:.2f}")
    else:
        st.warning("Please enter a claim into the text box first.")

st.markdown(
    '<div class="footer-note muted">This tool is for educational use only and should not be used as medical advice.</div>',
    unsafe_allow_html=True,
)
