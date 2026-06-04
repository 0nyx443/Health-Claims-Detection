# 🚀 Developer Guide: How to Run the Project Locally

This guide explains how to set up and run the **Health Claim Fact-Checker** application on your local machine.

---

## 📋 Prerequisites

Ensure you have **Python 3.8 or higher** installed on your machine. You can verify this by running:
```bash
python --version
```

---

## ⚡ Quick Start: Running the Streamlit App

Since the pre-trained vector mappings and the finalized machine learning model files (`glove.6B.50d.txt` and `health_claim_model.pkl`) are already committed to the repository, you can launch the app immediately without retraining.

### 1. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/0nyx443/Health-Claims-Detection.git
cd Health-Claims-Detection
```

### 2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies:

*   **On Windows (Command Prompt / PowerShell):**
    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```
*   **On macOS / Linux (Terminal):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 3. Install Required Dependencies
Install the required scientific and NLP libraries:
```bash
pip install pandas numpy scikit-learn nltk streamlit joblib requests
```

### 4. Run the Application
Start the Streamlit local development server:
```bash
streamlit run app.py
```
This will automatically open a tab in your web browser pointing to:
👉 **Local URL**: [http://localhost:8501](http://localhost:8501)

---

## 🔄 Optional: Retraining the Model

If you make modifications to the training dataset (`train.tsv`) or want to adjust the model parameters:

1. Run the training script:
   ```bash
   python train_model.py
   ```
2. The script will:
   * Perform text cleaning (lowercasing, punctuation stripping, lemmatization).
   * Evaluate the Linear Support Vector Classifier (LinearSVC) on a 30% test split.
   * Retrain on 100% of the dataset.
   * Save the newly updated `health_claim_model.pkl` and `tfidf_vectorizer.pkl` files.
3. Refresh your Streamlit browser tab to run predictions using your newly trained model.
