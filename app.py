import streamlit as st
import nltk
import joblib
import string
import os
import torch
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

# Preprocessing for baseline SVM fallback
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text_svm(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

# Static Knowledge Base for high-frequency direct health facts and myths
KNOWN_CLAIMS = {
    # Medical Facts (True)
    "eating fruits and vegetables provides important vitamins and minerals": "true",
    "smoking increases the risk of cancer and lung disease": "true",
    "washing your hands helps prevent the spread of infections": "true",
    "too much sugar intake can increase the risk of obesity and tooth decay": "true",
    "wearing sunscreen helps reduce the risk of skin cancer": "true",
    "exercising regularly strengthens the cardiovascular system": "true",
    "drinking enough water is essential for kidney function": "true",
    "antibiotics cure bacterial infections but do not work on viruses": "true",
    "high blood pressure increases the risk of heart disease and stroke": "true",
    "a balanced diet supports a healthy immune system": "true",
    
    # Medical Myths (False)
    "drinking lemon water cures cancer": "false",
    "vaccines cause autism": "false",
    "detox teas remove toxins from your body": "false",
    "microwave ovens make food radioactive": "false",
    "apple cider vinegar melts belly fat without diet or exercise": "false",
    "eating fat makes you fat instantly": "false",
    "organic food is always 100 pesticidefree": "false",
    "organic food is always 100 pesticide free": "false",
    "shaving makes hair grow back thicker and faster": "false",
    "cracking your knuckles causes arthritis": "false",
    "cold weather causes the common cold": "false"
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

# Clean and normalize input for exact static-match checking
normalized_input = str(user_input).strip().lower().translate(str.maketrans('', '', string.punctuation))

# Display model status indicators
if user_input and normalized_input in KNOWN_CLAIMS:
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
            
            # Step A: Check static knowledge base first for established medical consensus
            if normalized_input in KNOWN_CLAIMS:
                prediction = KNOWN_CLAIMS[normalized_input]
            else:
                # Step B: Pass to ML/DL models
                if loaded_model['type'] == 'distilbert':
                    tokenizer = loaded_model['tokenizer']
                    model = loaded_model['model']
                    
                    # Tokenize input claim
                    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
                    
                    # Forward pass without calculating gradients
                    with torch.no_grad():
                        outputs = model(**inputs)
                        
                    logits = outputs.logits
                    pred_idx = torch.argmax(logits, dim=1).item()
                    
                    # Map back to labels
                    labels = ['true', 'false', 'mixture', 'unproven']
                    prediction = labels[pred_idx]
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