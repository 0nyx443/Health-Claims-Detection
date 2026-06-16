import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)

def main():
    print("1. Checking hardware acceleration...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("\n⚠️  WARNING: No GPU detected. Local training on CPU may take hours.")
        print("💡 Recommended: Use the 'train_colab.ipynb' notebook on Google Colab for free GPU training (~5-10 minutes).\n")
    
    print("2. Loading dataset...")
    dataset_file = "train_balanced.tsv" if os.path.exists("train_balanced.tsv") else "train.tsv"
    print(f"Loading data from: {dataset_file}")
    df = pd.read_csv(dataset_file, sep='\t').dropna(subset=['claim', 'label']).rename(columns={'claim': 'text'})
    valid_labels = ['true', 'false', 'mixture', 'unproven']
    train_df = df[df['label'].isin(valid_labels)].copy()
    print(f"Dataset loaded. Total clean samples: {len(train_df)}")

    # Map string labels to numeric classes
    label_map = {'true': 0, 'false': 1, 'mixture': 2, 'unproven': 3}
    train_df['label_idx'] = train_df['label'].map(label_map)
    print("Class distribution:")
    print(train_df['label'].value_counts())

    print("3. Splitting dataset for evaluation (70% train / 30% test)...")
    X_train_texts, X_test_texts, y_train_labels, y_test_labels = train_test_split(
        train_df['text'].values,
        train_df['label_idx'].values,
        test_size=0.3,
        random_state=42,
        stratify=train_df['label_idx'].values
    )
    print(f"Training split: {len(X_train_texts)} samples")
    print(f"Testing split: {len(X_test_texts)} samples")

    print("4. Loading DistilBERT tokenizer and tokenizing text...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Using sequence length of 128 to save memory and compute time
    train_encodings = tokenizer(list(X_train_texts), truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(list(X_test_texts), truncation=True, padding=True, max_length=128)

    # PyTorch Dataset wrapper
    class HealthClaimsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = HealthClaimsDataset(train_encodings, y_train_labels)
    test_dataset = HealthClaimsDataset(test_encodings, y_test_labels)

    print("5. Initializing DistilBERT model for sequence classification...")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Define Hugging Face TrainingArguments and Trainer
    # On CPU we will restrict logging/checkpointing overhead
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("6. Starting training...")
    trainer.train()

    print("7. Generating evaluation report...")
    eval_results = trainer.predict(test_dataset)
    predictions = np.argmax(eval_results.predictions, axis=1)

    print("\n=== FINAL EVALUATION METRICS ===")
    print(classification_report(y_test_labels, predictions, target_names=valid_labels))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_labels, predictions))
    print("================================\n")

    output_dir = './distilbert_health_model'
    print(f"8. Saving trained model and tokenizer to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("SUCCESS: The final DistilBERT model and tokenizer have been saved.")

if __name__ == '__main__':
    main()