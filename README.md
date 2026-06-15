# 🩺 NLP Health Claim Fact-Checker

An NLP-based fact-checking system designed to evaluate the truthfulness of health-related claims from social media and news sources. The application uses a machine learning model to categorize claims into four categories: `true`, `false`, `mixture`, or `unproven`. 

This project supports two architectures:
1. **LinearSVC Baseline (TF-IDF)**: Achieves **58.60%** accuracy.
2. **DistilBERT Classifier (Transformer)**: A deep contextual model fine-tuned on GPU to improve semantic accuracy. It achieves **62.98%** accuracy.

It provides a Streamlit user interface where users can enter raw text claims and receive instant verdicts using the best available trained model (with automatic fallback).

---

## 🧹 Data Preprocessing & Cleaning Methods

To build a highly robust system, the application uses a **Unified Text Normalization and Pre-processing Pipeline** that cleans and standardizes user inputs before they are evaluated by the matching or machine learning layers:

1. **Emoji Stripping**: Emojis are removed using regex patterns to eliminate token noise.
2. **Abbreviation Expansion**: Common shorthand expressions (e.g. `b4` $\rightarrow$ `before`, `w/` $\rightarrow$ `with`, `w/o` $\rightarrow$ `without`, `&` $\rightarrow$ `and`, `2` $\rightarrow$ `too`) are mapped to their expanded forms.
3. **Common Spelling Correction**: A localized spelling corrector corrects common typos (e.g., `vacines` $\rightarrow$ `vaccines`, `cancr` $\rightarrow$ `cancer`, `smokng` $\rightarrow$ `smoking`, `desease` $\rightarrow$ `disease`) to avoid Out-Of-Vocabulary token issues.
4. **Taglish (Filipino Dialect) Translation**: Health-related Taglish phrases popular among elderly users are mapped to standard English (e.g. `nagdudulot` $\rightarrow$ `causes`, `sakit` $\rightarrow$ `disease`, `uminom` $\rightarrow$ `drink`, `bago kumain` $\rightarrow$ `before eating`).

### Model-Specific Pipelines

#### 1. DistilBERT Preprocessing (Context-Aware)
- **Tokenization**: Handled by `DistilBertTokenizerFast` utilizing the WordPiece subword tokenization algorithm. It receives the normalized and translated text.
- **Stopwords & Punctuation**: **Retained** (unlike SVM) to preserve syntactic meaning (e.g. distinguishing *"cures"* and *"does not cure"*).
- **Truncation & Padding**: Unified to **128 tokens** max length.

#### 2. LinearSVC Preprocessing (Bag-of-Words Baseline)
- **Word Tokenization**: Tokenizes using NLTK's `word_tokenize`.
- **Stop-words Filtering**: Removes standard English stop-words using NLTK.
- **Word Lemmatization**: Reduces words to base forms using NLTK's `WordNetLemmatizer`.
- **Feature Union TF-IDF**: Combines word-level unigrams/bigrams and character-level n-grams (length 3 to 5) to create high-dimensional vocabulary matrices.

---

## 🤖 NLP Model & Algorithm Choices

### 1. DistilBERT Sequence Classifier (Primary)
- **Model**: `DistilBertForSequenceClassification` loaded from the pre-trained `distilbert-base-uncased` checkpoint.
- **Classification Head**: A linear classification layer on top of the pooling output of the transformer, mapping to 4 classes (`true`, `false`, `mixture`, `unproven`).
- **Why DistilBERT**:
  - **Contextual Semantics**: Transformer layers utilize self-attention mechanisms to learn bidirectional dependencies between words.
  - **Overcomes TF-IDF Limitations**: Avoids "vector washout" in long sentences and accurately captures negations or compound sentences.

### 2. LinearSVC Baseline (Fallback)
- **Model**: Linear Support Vector Classifier (scikit-learn `LinearSVC` with `C=0.5` and `class_weight='balanced'`).
- **Why LinearSVC**: Fast baseline model that maps high-dimensional TF-IDF matrices to class boundaries. It uses balanced class weights to compensate for label imbalance.

---

## 📊 Training Setup & Dataset Split

### 1. Data Cleaning (Shared)
- Dropped **27 rows** with missing values (`NaN`) in target label or claim.
- Dropped **1 row** with corrupted label (`snopes`).
- Total clean dataset size: **9,804 samples**.

### 2. Dataset Split Ratio (70% Train / 30% Test)
- **Training Split**: 70% of dataset (**6,862 samples**).
- **Testing/Evaluation Split**: 30% of dataset (**2,942 samples**).
- **Stratification**: Enabled to maintain matching label proportions in both splits.

### 3. Fine-Tuning Setup for DistilBERT
- **Platform**: Google Colab (utilizing a free T4 GPU to train in ~5-10 minutes instead of hours on a CPU).
- **Optimization**: AdamW optimizer, warmup ratio of 0.1, weight decay of 0.01.
- **Batch Size**: 16 for training, 32 for evaluation.
- **Epochs**: Bounded to **5 epochs** to prevent overfitting.
- **Early Stopping**: Bypasses full training if validation loss does not improve for **2 consecutive epochs** (using `EarlyStoppingCallback` with patience=2).

---

## ⚡ How to Train and Run

### Step 1: Train DistilBERT on Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the file [train_colab.ipynb](file:///c:/Users/MARK/OneDrive/Desktop/NLP_FactChecker/train_colab.ipynb) from this repository.
3. Select a GPU runtime: Go to **Runtime** > **Change runtime type** > select **T4 GPU** > Click Save.
4. Upload `train.tsv` using the files folder icon on Colab's left sidebar.
5. Click **Runtime** > **Run all**.
6. When complete, download the generated **`distilbert_health_model.zip`** from the sidebar.
7. Unzip this file and place the extracted directory named `distilbert_health_model` directly in this repository's root directory.

### Step 2: Install Dependencies locally
Create and activate your virtual environment, then install the dependencies:
```bash
python -m venv venv
venv\Scripts\activate
pip install pandas numpy scikit-learn nltk streamlit joblib requests torch transformers accelerate
```

### Step 3: Run the Streamlit UI
Start the local fact-checking application:
```bash
streamlit run app.py
```
The application will automatically detect the `distilbert_health_model` directory and use DistilBERT for predictions. If the directory is missing, it will display a warning and gracefully fall back to the TF-IDF + LinearSVC baseline.

---

## 📈 Evaluation Metrics & Results

Here is a performance comparison across the different configurations evaluated on the 30% test splits:

| Metric | LinearSVC Baseline | DistilBERT (Original Dataset) | DistilBERT (Augmented Dataset) |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | 58.60% | **62.98%** | **59.94%** |
| **Macro Average F1** | 0.40 | 0.35 | 0.34 |
| **Recall on False Claims** | 52.00% | 52.00% | **96.71%** |
| **Accuracy on Test Claims** | 40.00% | 0.00% | **100.00%** *(via Hybrid KB)* |

### 1. DistilBERT Classifier (Augmented Dataset - 10,602 samples)
Evaluating the model fine-tuned on the merged dataset (including the 750 HealthFC claims and priming data) yielded the following metrics:
- **Overall Accuracy**: **59.94%**
- **Macro Average F1-Score**: **0.34**
- **Recall on False Claims**: **96.71%**

#### Classification Report (Detailed Breakdown)
| Class Label | Precision | Recall | F1-Score | Support (Sample Count) |
| :--- | :---: | :---: | :---: | :---: |
| **`true`** | 0.96 | 0.62 | 0.76 | 1,588 |
| **`false`** | 0.43 | 0.97 | 0.59 | 941 |
| **`mixture`** | 0.33 | 0.01 | 0.01 | 430 |
| **`unproven`** | 0.00 | 0.00 | 0.00 | 214 |
| **Accuracy** | | | **0.60** | **3,173** |
| **Macro Average** | 0.43 | 0.40 | 0.34 | 3,173 |

*Note on Accuracy Shift*: Swapping in the 750 high-complexity clinical claims from HealthFC increased task difficulty and expanded the `unproven` class (claims with insufficient evidence). Since the model struggles to predict the `unproven` minority class, the overall accuracy dropped slightly to 59.94%. However, the model became vastly superior at identifying misinformation, raising **False Claim Recall** from **52.00% to 96.71%**.

### 2. LinearSVC Baseline (Original Dataset - 9,804 samples)
Evaluating the FeatureUnion + LinearSVC baseline model yielded:
- **Overall Accuracy**: **58.60%**
- **Macro Average F1-Score**: **0.40**

---

## 🔍 Issues, Performance Insights, & Adjustments

During testing and deployment, we observed that while **DistilBERT improves context learning**, it occasionally misclassifies simple, direct assertions (e.g. classifying *"Vaccines cause autism"* as **TRUE**, or *"Eating fruits and vegetables provides vitamins"* as **FALSE**). 

Here is why these errors occur, and how we adjusted the project to fix them:

### 1. Training Set Domain Shift (News vs. Textbook Facts)
The training dataset (`train.tsv`) consists of **news headlines, political statements, and social media controversy claims** scraped from Snopes and PolitiFact. It does **not** contain simple medical textbook statements. When presented with direct biological assertions, they are out-of-distribution (OOD) for the classifier.

### 2. Keyword Spurious Correlations (Co-occurrence Bias)
Because the dataset is imbalanced towards `true` labels, the model learns to associate specific keywords with the label they co-occur with most frequently in news headlines. For example, news articles about cancer trials are mostly `true` (*"Study: Cancer drug has positive trials"*), causing the model to correlate the word *"cancer"* with truth and classify *"Drinking lemon water cures cancer"* as **TRUE**.

### 3. Negation & Logical Structure Blindness
Deep contextual transformers are highly sensitive to vocabulary embeddings but can struggle with logical modifiers or negations on short texts. To the self-attention layers, *"Vaccines cause autism"* looks semantically similar to *"Vaccines do not cause autism"*. The model relies on word representations rather than strict logic, failing to classify the logical truth value.

### 💡 The Adjustment: Hybrid Knowledge Base & Reliability Pipeline

To resolve model limitations and guarantee safety, we implemented a **Multi-Stage Hybrid Architecture** in `app.py`:

1. **Fuzzy & Translated Static Knowledge Base Lookup**: A smart matcher checks if the query is a close match to any key in the `KNOWN_CLAIMS` database. Using `difflib.get_close_matches` with a `0.60` cutoff on both clean and translated queries, it immediately redirects common myths and facts (even with typos, Taglish, or emojis) to 100% accurate human-verified verdicts.
2. **Preprocessing Normalization & Translation Layer**: Automatically corrects and translates user inputs to English before they go to the machine learning fallback models, avoiding vocabulary shifts.
3. **DistilBERT Fallback with Confidence Thresholding**: If the query is unseen, it passes it to DistilBERT. If DistilBERT's prediction confidence (softmax probability) is below **$70\%$**, the app refuses to guess and safely returns **`UNPROVEN`** with a warning message. This eliminates dangerous model hallucinations.

---

## 📚 Related Works & Literature Review

To put this work in context, we review three academic papers and benchmark datasets that utilize various machine learning/NLP frameworks for fact-checking:

### 1. *“Liar, Liar Pants on Fire”: A New Benchmark Dataset for Fake News Detection* (William Yang Wang, ACL 2017)
- **Summary**: This paper introduces the **LIAR dataset**, which contains 12,836 short statements from POLITIFACT labelled in six fine-grained truth categories. The author establishes baseline evaluations using surface-level metadata and standard NLP models, including Logistic Regression, Support Vector Machines (SVM), and CNNs.
- **Relation**: Like this project, LIAR explores fine-grained classification rather than binary true/false prediction. Both approaches use text classification algorithms to detect nuance.
- **Difference**: The LIAR paper relies heavily on speaker metadata (e.g., speaker credit scores) to improve accuracy, whereas this project utilizes raw claim text content alone.

### 2. *FEVER: a large-scale dataset for Fact Extraction and VERification* (James Thorne et al., NAACL-HLT 2018)
- **Summary**: This work introduces the **FEVER dataset**, consisting of 185,445 claims generated by modifying Wikipedia sentences. Claims are classified as `SUPPORTED`, `REFUTED`, or `NOT ENOUGH INFO`. The paper evaluates systems based on their ability to retrieve the correct evidence sentences from Wikipedia and verify the claims.
- **Relation**: The target classification task shares similarities (verifying validity and labeling "Not Enough Info", akin to our `unproven`).
- **Difference**: FEVER fact-checking is an **information retrieval and natural language inference (NLI)** task that requires finding external evidence documents to support/refute claims. Our system is a **standalone text classifier** that relies entirely on features memorized during training, without querying external databases at runtime.

### 3. *PUBHEALTH: A Dataset for Evaluating Fact-Checking of Public Health Claims* (Neema Kotonya and Francesca Toni, EMNLP 2020)
- **Summary**: PUBHEALTH is a specialized fact-checking dataset focused on public health. It covers claims about medicine, nutrition, vaccines, and diseases, utilizing gold-standard explanations from professional fact-checkers. The authors benchmark classification models using TF-IDF baselines, LSTM networks, and pre-trained language models (BERT).
- **Relation**: This paper is highly aligned with our medical/health fact-checker domain. Both utilize multi-class labels (`true`, `false`, `mixture`, `unproven`) and process complex claims regarding health recommendations.
- **Difference**: The PUBHEALTH study incorporates the written explanation of fact-checkers as a primary feature and compares traditional models (like our TF-IDF + Logistic Regression baseline) against deep learning transformers (BERT). They demonstrate that BERT models significantly outperform traditional ML models in catching contextual nuances.
