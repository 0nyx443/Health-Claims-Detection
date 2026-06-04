# 🩺 NLP Health Claim Fact-Checker

An NLP-based fact-checking system designed to evaluate the truthfulness of health-related claims from social media and news sources. The application uses a machine learning model to categorize claims into four categories: `true`, `false`, `mixture`, or `unproven`. It provides a Streamlit user interface where users can paste raw text claims and receive instant verdicts.

---

## 🧹 Data Preprocessing & Cleaning Methods

To prepare raw textual health claims for machine learning, the following pipeline is executed sequentially:

1. **Case Normalization**: All characters are converted to lowercase. This ensures words like "Cancer", "CANCER", and "cancer" are represented identically.
2. **Punctuation Removal**: Special characters and punctuation marks are stripped out (using `string.punctuation`) since they do not contribute to semantic classification.
3. **Word Tokenization**: Sentences are segmented into individual word tokens using NLTK's `word_tokenize`.
4. **Stop-Words Filtering**: Highly frequent but low-information words (such as *"the"*, *"is"*, *"in"*, *"and"*) are removed using NLTK's English stop-words dictionary.
5. **Word Lemmatization**: Words are reduced to their base dictionary form (lemma) using the `WordNetLemmatizer` (e.g., *"leaves"* $\rightarrow$ *"leaf"*, *"cures"* $\rightarrow$ *"cure"*).
6. **Feature Union (Word-level + Character-level TF-IDF)**:
   * **Word Vectorizer**: Extracts word-level unigrams and bigrams (`ngram_range=(1, 2)`) with **`max_features=None`** (to utilize the entire vocabulary of the dataset). This captures phrase-level indicators (like *"can cure"*, *"poor test"*).
   * **Character Vectorizer**: Extracts character-level n-grams of length 3 to 5 (`analyzer='char_wb', ngram_range=(3, 5)`) with **`max_features=None`** (to capture all sub-parts, prefixes, and suffixes in the dataset). This captures roots, prefixes, and suffixes (e.g., *"arthr"*, *"ritis"*, *"vacc"*, *"garlic"*).
   * **Combined Features**: Combines both vectorizers into a single, high-dimensional representation of 100% of the dataset's vocabulary without artificial limits.

---

## 🤖 NLP Model & Algorithm Choice

* **Text Representation**: FeatureUnion of Word-level and Character-level TF-IDF.
* **Classifier**: **Linear Support Vector Classifier (LinearSVC)** (`C=0.5`, `class_weight='balanced'`, `random_state=42`).
* **Why Word+Char TF-IDF + LinearSVC**:
  * **No Vector Washout**: Unlike word vector averaging (which averages vectors together and dilutes signals in long claims), TF-IDF keeps word and character frequencies discrete. This allows specific key phrases or roots to retain their full predictive power.
  * **Out-of-Vocabulary Matching**: By extracting character-level n-grams, the model can recognize suffixes, prefixes, and roots of unseen words. If a new word (like *"knuckles"* or *"garlic"*) appears at runtime, it matches sub-parts of words the model has learned, resolving out-of-vocabulary limits.
  * **Balanced Generalization**: The model uses a Support Vector Classifier with balanced class weights, which scales the margins to prevent majority class bias towards `true` claims.

---

## 📊 Training Setup & Dataset Split

1. **Cleaning Anomalies**: 
   * Dropped **27 rows** with missing values (`NaN`) in the target label or claim text.
   * Dropped **1 row** with a corrupted label (`snopes`).
   * Total clean dataset size: **9,804 samples**.
2. **Train/Test Evaluation Split (70/30)**:
   * **Training Split**: 70% of the dataset (**6,862 samples**) was used to train the classifier.
   * **Testing Split**: 30% of the dataset (**2,942 samples**) was reserved for evaluation.
   * **Stratification**: Enabled to maintain the exact class distribution across splits.
3. **Full Production Retraining**: The final production vectorizer pipeline and classifier were retrained on **100% of the cleaned dataset (9,804 samples)** to maximize prediction accuracy in the Streamlit app.

---

## 📈 Evaluation Metrics & Results

Evaluating the FeatureUnion + LinearSVC model on the 30% test split (2,942 unseen samples) yielded the following metrics:

* **Overall Accuracy**: **58.60%**
* **Macro Average F1-Score**: **0.40**

### Classification Report (Detailed Breakdown)

| Class Label | Precision | Recall | F1-Score | Support (Sample Count) |
| :--- | :---: | :---: | :---: | :---: |
| **`true`** | 0.72 | 0.76 | 0.74 | 1,524 |
| **`false`** | 0.49 | 0.52 | 0.50 | 901 |
| **`mixture`** | 0.26 | 0.20 | 0.23 | 430 |
| **`unproven`** | 0.26 | 0.09 | 0.14 | 87 |
| **Accuracy** | | | **0.59** | **2,942** |
| **Macro Average** | 0.43 | 0.39 | 0.40 | 2,942 |
| **Weighted Average** | 0.57 | 0.59 | 0.58 | 2,942 |

### Confusion Matrix

The confusion matrix rows represent the actual labels and columns represent the predicted labels, in the order `['true', 'false', 'mixture', 'unproven']`:

$$\begin{pmatrix}
1159 & 245 & 115 & 5 \\
282 & 469 & 136 & 14 \\
138 & 200 & 88 & 4 \\
23 & 50 & 6 & 8
\end{pmatrix}$$

---

## 🔍 Issues, Performance Insights, & Adjustments

During testing of the word vector models, we observed a "vector washout" effect where averaging vectors on long, unseen sentences pulled them toward the center of the vector space, resulting in predictions of `MIXTURE` or `UNPROVEN` for almost all claims.
* **The Adjustment**:
  1. We replaced embeddings with a **Word-level and Character-level TF-IDF Feature Union**, which keeps features discrete and captures word roots (like *"arthr"*, *"ritis"*, *"vacc"*).
  2. We configured the vectorizers to use **`max_features=None`**, utilizing 100% of the dataset's vocabulary features.
  3. We trained a **LinearSVC** model with balanced weights. 
* **The Results**:
  * Accuracy restored to a robust **58.60%**.
  * The model successfully classifies **5 out of 6 of our dataset test claims**.
  * On new, out-of-dataset claims, the character-level features correctly identify major health myths (e.g. *"Vaccines cause autism"* $\rightarrow$ **FALSE**, and *"Garlic paste cures open wound infections"* $\rightarrow$ **FALSE**).

---

## 📚 Related Works & Literature Review

To put this work in context, we review three academic papers and benchmark datasets that utilize various machine learning/NLP frameworks for fact-checking:

### 1. *“Liar, Liar Pants on Fire”: A New Benchmark Dataset for Fake News Detection* (William Yang Wang, ACL 2017)
* **Summary**: This paper introduces the **LIAR dataset**, which contains 12,836 short statements from POLITIFACT labelled in six fine-grained truth categories. The author establishes baseline evaluations using surface-level metadata and standard NLP models, including Logistic Regression, Support Vector Machines (SVM), and CNNs.
* **Similarities**: Like this project, LIAR explores fine-grained classification rather than binary true/false prediction. Both approaches use text classification algorithms to detect nuance.
* **Differences**: The LIAR paper relies heavily on speaker metadata (e.g., speaker credit scores) to improve accuracy, whereas this project utilizes raw claim text content alone.

### 2. *FEVER: a large-scale dataset for Fact Extraction and VERification* (James Thorne et al., NAACL-HLT 2018)
* **Summary**: This work introduces the **FEVER dataset**, consisting of 185,445 claims generated by modifying Wikipedia sentences. Claims are classified as `SUPPORTED`, `REFUTED`, or `NOT ENOUGH INFO`. The paper evaluates systems based on their ability to retrieve the correct evidence sentences from Wikipedia and verify the claims.
* **Similarities**: The target classification task shares similarities (verifying validity and labeling "Not Enough Info", akin to our `unproven`).
* **Differences**: FEVER fact-checking is an **information retrieval and natural language inference (NLI)** task that requires finding external evidence documents to support/refute claims. Our system is a **standalone text classifier** that relies entirely on features memorized during training, without querying external databases at runtime.

### 3. *PUBHEALTH: A Dataset for Evaluating Fact-Checking of Public Health Claims* (Neema Kotonya and Francesca Toni, EMNLP 2020)
* **Summary**: PUBHEALTH is a specialized fact-checking dataset focused on public health. It covers claims about medicine, nutrition, vaccines, and diseases, utilizing gold-standard explanations from professional fact-checkers. The authors benchmark classification models using TF-IDF baselines, LSTM networks, and pre-trained language models (BERT).
* **Similarities**: This paper is highly aligned with our medical/health fact-checker domain. Both utilize multi-class labels (`true`, `false`, `mixture`, `unproven`) and process complex claims regarding health recommendations.
* **Differences**: The PUBHEALTH study incorporates the written explanation of fact-checkers as a primary feature and compares traditional models (like our TF-IDF + Logistic Regression baseline) against deep learning transformers (BERT). They demonstrate that BERT models significantly outperform traditional ML models in catching contextual nuances.
