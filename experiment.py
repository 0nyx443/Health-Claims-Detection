import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import string

print("1. Loading dataset...")
df = pd.read_csv('train.tsv', sep='\t').dropna(subset=['claim', 'label']).rename(columns={'claim': 'text'})
valid_labels = ['true', 'false', 'mixture', 'unproven']
train_df = df[df['label'].isin(valid_labels)].copy()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

print("2. Cleaning text...")
train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)

print("3. Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    train_df['cleaned_text'], 
    train_df['label'], 
    test_size=0.3, 
    random_state=42, 
    stratify=train_df['label']
)

test_claims = [
    "Drinking lemon juice can cure cancer",
    "Herbal remedies or specific diets can treat, prevent, or 'cure' serious illnesses like cancer, diabetes, or Alzheimer's.",
    "Certain proprietary supplement blends can 'instantly' cure or prevent the common cold and flu."
]

def evaluate_model(name, tfidf, model):
    print(f"\n==================== {name} ====================")
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, labels=valid_labels, target_names=valid_labels))
    
    print("Predictions on sample claims:")
    for claim in test_claims:
        cleaned = preprocess_text(claim)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        print(f"  Claim: '{claim}'")
        print(f"  Predicted: {pred.upper()}")

# Experiment 1: Baseline Random Forest
evaluate_model(
    "Baseline: Random Forest (No Class Balance)", 
    TfidfVectorizer(max_features=5000), 
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
)

# Experiment 2: Logistic Regression with Class Balancing
evaluate_model(
    "Logistic Regression (class_weight='balanced')", 
    TfidfVectorizer(ngram_range=(1, 2), max_features=10000), 
    LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
)

# Experiment 3: Linear Support Vector Classification (LinearSVC) with Class Balancing
evaluate_model(
    "Linear Support Vector Classification (class_weight='balanced')", 
    TfidfVectorizer(ngram_range=(1, 2), max_features=10000), 
    LinearSVC(random_state=42, class_weight='balanced')
)

# Experiment 4: Multinomial Naive Bayes
evaluate_model(
    "Multinomial Naive Bayes (No weights)", 
    TfidfVectorizer(ngram_range=(1, 2), max_features=10000), 
    MultinomialNB()
)
