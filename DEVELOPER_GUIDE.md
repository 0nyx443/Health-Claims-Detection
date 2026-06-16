# 🚀 Developer Guide: How to Run the Project Locally

This guide explains how to set up, run, and retrain the **Health Claim Fact-Checker** application on your local machine.

---

## 📋 Prerequisites

Ensure you have **Python 3.8 or higher** installed on your machine. You can verify this by running:
```bash
python --version
```

---

## ⚡ Quick Start: Running the Streamlit App

The project includes an automatic fallback system. If the advanced DistilBERT model is not trained/downloaded, it will automatically fall back to the pre-trained `LinearSVC` baseline model.

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
Install the required packages, including PyTorch and Hugging Face transformers:
```bash
pip install pandas numpy scikit-learn nltk streamlit joblib requests torch transformers accelerate
```

### 4. Run the Application
Start the Streamlit local development server:
```bash
streamlit run app.py
```
This will automatically open a tab in your web browser pointing to:
👉 **Local URL**: [http://localhost:8501](http://localhost:8501)

If you have not downloaded the DistilBERT model, the UI will display:
`Active Model: LinearSVC (TF-IDF Baseline)` along with a reminder on how to upgrade.

---

## 🔄 Retraining the Models

### Option A: Fine-Tuning DistilBERT on Google Colab (Recommended)
Training transformer models on a CPU takes hours. We recommend using a free GPU on Google Colab:
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the notebook file [train_colab.ipynb](file:///c:/Users/MARK/OneDrive/Desktop/NLP_FactChecker/train_colab.ipynb) from this repository.
3. Select a GPU runtime: Go to **Runtime** > **Change runtime type** > Select **T4 GPU** > Click Save.
4. Upload `train.tsv` to the file explorer sidebar.
5. Click **Runtime** > **Run all**.
6. When complete, download the generated **`distilbert_health_model.zip`** from the file explorer sidebar.
7. Unzip this file and place the extracted directory named `distilbert_health_model` directly in this repository's root directory.
8. Refresh your Streamlit browser tab to run predictions using your newly trained DistilBERT transformer!

### Option B: Local Training Fallback (LinearSVC baseline or CPU DistilBERT)
If you make modifications to the training dataset (`train.tsv`) or want to run training locally:
1. Run the local training script:
   ```bash
   python train_model.py
   ```
2. The script will:
   - Check if a local GPU is available (falls back to CPU if not).
   - Split the dataset into 70% train / 30% test.
   - Fine-tune DistilBERT for 5 epochs with early stopping (patience=2).
   - Save the model weights to the `distilbert_health_model` folder.
