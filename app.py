import streamlit as st
import nltk
import joblib
import string
import os
import torch
import re
import difflib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

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

# Helper for Static KB Lookup with Fuzzy and Translation matching
def check_static_kb(text):
    cleaned = clean_and_normalize_text(text)
    # Strip standard punctuation for lookup comparison
    cleaned_no_punct = cleaned.translate(str.maketrans('', '', string.punctuation)).strip()
    translated = translate_taglish_to_english(cleaned_no_punct).strip()
    
    # 1. Exact match on normalized or translated
    if cleaned_no_punct in KNOWN_CLAIMS:
        return KNOWN_CLAIMS[cleaned_no_punct], cleaned_no_punct
    if translated in KNOWN_CLAIMS:
        return KNOWN_CLAIMS[translated], translated
        
    # 2. Fuzzy match on normalized (cutoff 0.60)
    matches_cleaned = difflib.get_close_matches(cleaned_no_punct, KNOWN_CLAIMS.keys(), n=1, cutoff=0.60)
    if matches_cleaned:
        return KNOWN_CLAIMS[matches_cleaned[0]], matches_cleaned[0]
        
    # 3. Fuzzy match on translated (cutoff 0.60)
    matches_translated = difflib.get_close_matches(translated, KNOWN_CLAIMS.keys(), n=1, cutoff=0.60)
    if matches_translated:
        return KNOWN_CLAIMS[matches_translated[0]], matches_translated[0]
        
    return None, None


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
        try:
            tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_DIR)
            model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_DIR)
            # Put in evaluation mode
            model.eval()
            return {
                'type': 'distilbert',
                'model': model,
                'tokenizer': tokenizer
            }
        except Exception as e:
            st.warning(f"Failed to load DistilBERT from {DISTILBERT_DIR}. Error: {e}. Falling back to baseline SVM model.")
    
    # Load fallback SVM model
    if os.path.exists(SVM_MODEL_PATH) and os.path.exists(SVM_VECTORIZER_PATH):
        try:
            model = joblib.load(SVM_MODEL_PATH)
            vectorizer = joblib.load(SVM_VECTORIZER_PATH)
            return {
                'type': 'svm',
                'model': model,
                'vectorizer': vectorizer
            }
        except Exception as e:
            st.error(f"Failed to load SVM baseline. Error: {e}")
            st.stop()
    else:
        st.error("No classification model found! Please train the model using Google Colab or train_model.py first.")
        st.stop()

# Load the available model
loaded_model = load_fact_checker_model()

# 3. Modern Minimalist CSS with Plus Jakarta Sans & Light Palette
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* Apply font and background to Streamlit container */
.stApp {
    background-color: #f8fafc;
}

html, body, [class*="css"], .stMarkdown {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.title-container {
    padding-top: 40px;
    margin-bottom: 20px;
    text-align: center;
}

.title-text {
    font-size: 36px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
}

.subtitle-text {
    font-size: 16px;
    color: #64748b;
    margin-bottom: 15px;
}

/* Badge styling for active model */
.model-badge {
    display: inline-block;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 600;
    border-radius: 9999px;
    margin-bottom: 20px;
}

.model-badge.distilbert {
    background-color: #e0f2fe;
    color: #0369a1;
    border: 1px solid #bae6fd;
}

.model-badge.svm {
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid #e2e8f0;
}

.model-badge.kb {
    background-color: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
}

/* Styled text area container */
.stTextArea textarea {
    border: 1px solid #cbd5e1 !important;
    background-color: #ffffff !important;
    border-radius: 8px !important;
    font-size: 15px !important;
    color: #0f172a !important;
    box-shadow: none !important;
    transition: border-color 0.2s ease !important;
}

.stTextArea textarea:focus {
    border-color: #64748b !important;
}

/* Styled Primary Button */
.stButton button {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    transition: background-color 0.2s ease, transform 0.1s ease !important;
}

.stButton button:hover {
    background-color: #1e293b !important;
}

.stButton button:active {
    transform: scale(0.98);
}

/* Results section card styling */
.verdict-card {
    padding: 24px;
    border-radius: 12px;
    margin: 24px 0;
    border: 1px solid #e2e8f0;
    background-color: #ffffff;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
}

.verdict-card.true {
    border-left: 6px solid #10b981;
    background-color: #f0fdf4;
}

.verdict-card.false {
    border-left: 6px solid #ef4444;
    background-color: #fef2f2;
}

.verdict-card.mixture {
    border-left: 6px solid #f59e0b;
    background-color: #fffbeb;
}

.verdict-card.unproven {
    border-left: 6px solid #64748b;
    background-color: #f1f5f9;
}

.verdict-header {
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

.verdict-header.true { color: #047857; }
.verdict-header.false { color: #b91c1c; }
.verdict-header.mixture { color: #b45309; }
.verdict-header.unproven { color: #475569; }

.verdict-title {
    font-size: 22px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 8px;
}

.verdict-desc {
    font-size: 15px;
    color: #334155;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# 4. Streamlit User Interface
st.markdown("""
<div class="title-container">
    <div class="title-text">Health Claim Fact-Checker</div>
    <div class="subtitle-text">Verify health-related assertions using machine learning and textual analysis</div>
</div>
""", unsafe_allow_html=True)

user_input = st.text_area("Enter Health Claim:", placeholder="e.g., Drinking boiled mango leaves cures hypertension...")

# Clean and normalize input for fuzzy static-match checking
kb_verdict, kb_matched_claim = check_static_kb(user_input)

# Display model status indicators
if user_input and kb_verdict is not None:
    st.markdown('<div class="model-badge kb">✅ Analysis Source: Verified Medical Knowledge Base</div>', unsafe_allow_html=True)
else:
    if loaded_model['type'] == 'distilbert':
        st.markdown('<div class="model-badge distilbert">⚡ Analysis Source: DistilBERT (Context-Aware Transformer)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="model-badge svm">📊 Analysis Source: LinearSVC (TF-IDF Baseline)</div>', unsafe_allow_html=True)

if loaded_model['type'] == 'svm':
    st.info(
        "💡 **Upgrade to DistilBERT**: Want better accuracy? Open the `train_colab.ipynb` file in Google Colab, "
        "train the transformer model on a free GPU, download the generated zip file, extract it to a "
        "folder named `distilbert_health_model` in your project folder, and refresh this page!"
    )

if st.button("Check Claim", type="primary"):
    if user_input:
        with st.spinner('Analyzing claim...'):
            prediction = None
            is_low_confidence = False
            confidence_score = 1.0
            
            # Step A: Check static knowledge base first for established medical consensus (using fuzzy match)
            kb_verdict, kb_matched_claim = check_static_kb(user_input)
            if kb_verdict is not None:
                prediction = kb_verdict
            else:
                # Step B: Pass to ML/DL models
                if loaded_model['type'] == 'distilbert':
                    tokenizer = loaded_model['tokenizer']
                    model = loaded_model['model']
                    
                    # Preprocess and translate input claim
                    normalized_input = clean_and_normalize_text(user_input)
                    translated_input = translate_taglish_to_english(normalized_input)
                    
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
                    
                    # Confidence thresholding to prevent false truths
                    if confidence_score < 0.70:
                        prediction = 'unproven'
                        is_low_confidence = True
                    else:
                        prediction = labels[pred_idx.item()]
                else:
                    # Baseline SVM prediction
                    svc_model = loaded_model['model']
                    tfidf = loaded_model['vectorizer']
                    
                    cleaned_input = preprocess_text_svm(user_input)
                    vectorized_input = tfidf.transform([cleaned_input])
                    prediction = svc_model.predict(vectorized_input)[0]
            
            # Display results
            st.divider()
            st.subheader("Analysis Result:")
            
            # Display match information or confidence metrics
            if kb_verdict is not None:
                st.info(f"📍 **Matches established medical claim**: \"*{kb_matched_claim}*\"")
            elif loaded_model['type'] == 'distilbert':
                if is_low_confidence:
                    st.warning(f"⚠️ **Low Confidence ({confidence_score:.1%})**: The model is uncertain about this assertion. Classifying as **Unproven** to prevent misinformation.")
                else:
                    st.success(f"🤖 **Model Confidence Score**: {confidence_score:.1%}")
            
            if prediction == "true":
                st.markdown(
                    '<div class="verdict-card true">'
                    '<div class="verdict-header true">Verdict</div>'
                    '<div class="verdict-title">True</div>'
                    '<div class="verdict-desc">This claim aligns with established medical facts.</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            elif prediction == "false":
                st.markdown(
                    '<div class="verdict-card false">'
                    '<div class="verdict-header false">Verdict</div>'
                    '<div class="verdict-title">False</div>'
                    '<div class="verdict-desc">Warning: This claim is flagged as potentially false or misleading.</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            elif prediction == "mixture":
                st.markdown(
                    '<div class="verdict-card mixture">'
                    '<div class="verdict-header mixture">Verdict</div>'
                    '<div class="verdict-title">Mixture</div>'
                    '<div class="verdict-desc">This claim has a mixed status, containing elements of both truth and error.</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else: # unproven
                st.markdown(
                    '<div class="verdict-card unproven">'
                    '<div class="verdict-header unproven">Verdict</div>'
                    '<div class="verdict-title">Unproven</div>'
                    '<div class="verdict-desc">This claim has an unproven status, with insufficient scientific evidence.</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
    else:
        st.warning("Please enter a claim into the text box first.")