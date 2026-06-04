import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import string

print("1. Loading NLTK resources...")
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

print("2. Loading dataset...")
df = pd.read_csv('train.tsv', sep='\t').dropna(subset=['claim', 'label']).rename(columns={'claim': 'text'})
valid_labels = ['true', 'false', 'mixture', 'unproven']
train_df = df[df['label'].isin(valid_labels)].copy()
print(f"Dataset loaded. Total clean samples: {len(train_df)}")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

print("3. Cleaning text (this might take a few seconds)...")
train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)

print("4. Splitting dataset for evaluation (70% train / 30% test)...")
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    train_df['cleaned_text'], 
    train_df['label'], 
    test_size=0.3, 
    random_state=42, 
    stratify=train_df['label']
)
print(f"Training split: {X_train_split.shape[0]} samples")
print(f"Testing split: {X_test_split.shape[0]} samples")

print("5. Evaluating model on split...")
# Combined Word n-grams and Char n-grams TF-IDF vectorizer (unlimited features)
word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=None)
char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=None)
union_eval = FeatureUnion([
    ('word', word_vec),
    ('char', char_vec)
])

# Vectorize splits
X_train_vec = union_eval.fit_transform(X_train_split)
X_test_vec = union_eval.transform(X_test_split)

# Train LinearSVC Classifier on split
svc_eval = LinearSVC(C=0.5, class_weight='balanced', random_state=42)
svc_eval.fit(X_train_vec, y_train_split)

# Predict and report metrics
y_pred = svc_eval.predict(X_test_vec)
accuracy = accuracy_score(y_test_split, y_pred)
print("\n=== MODEL EVALUATION METRICS (30% Test Split) ===")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_split, y_pred, labels=valid_labels, target_names=valid_labels))
print("Confusion Matrix:")
print(confusion_matrix(y_test_split, y_pred, labels=valid_labels))
print("=================================================\n")

print("6. Retraining final model on 100% of cleaned data (full utilization)...")
word_vec_final = TfidfVectorizer(ngram_range=(1, 2), max_features=None)
char_vec_final = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=None)
union_final = FeatureUnion([
    ('word', word_vec_final),
    ('char', char_vec_final)
])

X_full = union_final.fit_transform(train_df['cleaned_text'])
y_full = train_df['label']

svc_final = LinearSVC(C=0.5, class_weight='balanced', random_state=42)
svc_final.fit(X_full, y_full)

print("7. Saving final production files...")
joblib.dump(svc_final, 'health_claim_model.pkl')
joblib.dump(union_final, 'tfidf_vectorizer.pkl')
print("SUCCESS: The final LinearSVC model and FeatureUnion vectorizer have been saved and are ready for Streamlit.")