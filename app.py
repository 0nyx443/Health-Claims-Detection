import streamlit as st
import nltk
import joblib
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set page config without emojis
st.set_page_config(page_title="Health Fact Checker", page_icon=None, layout="centered")

# Download resources silently
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

# 1. Load the trained SVM model and the FeatureUnion TF-IDF vectorizer
@st.cache_resource
def load_models_and_vectorizer():
    model = joblib.load('health_claim_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    svc_model, tfidf = load_models_and_vectorizer()
except Exception as exc:
    st.error(
        "The app could not load the model or vectorizer pickle files. Please make sure train_model.py has run successfully."
    )
    st.exception(exc)
    st.stop()

# 2. Text Preprocessing Pipeline
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

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
    margin-bottom: 30px;
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

if st.button("Check Claim", type="primary"):
    if user_input:
        with st.spinner('Analyzing claim...'):
            # Preprocess and vectorize
            cleaned_input = preprocess_text(user_input)
            vectorized_input = tfidf.transform([cleaned_input])
            
            # Predict
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