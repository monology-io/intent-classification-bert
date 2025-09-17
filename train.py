import os
import json
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading libraries...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_intent_classifier(data_file="./data/sample_data.csv", output_dir="./intent_model"):
    """
    Train intent classification model using manual training loop.
    """
    
    # Load and validate data
    print(f"Loading dataset from {data_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset file {data_file} not found!")
    
    df = pd.read_csv(data_file)
    
    if 'query' not in df.columns or 'intent' not in df.columns:
        raise ValueError("CSV must have 'query' and 'intent' columns")
    
    # Clean data exactly like working Colab version
    df.dropna(subset=['query', 'intent'], inplace=True)
    df['intent'] = df['intent'].astype(str)
    df['query'] = df['query'].astype(str)
    df.reset_index(drop=True, inplace=True)
    
    if df['intent'].nunique() < 2:
        raise ValueError(f"Need at least 2 intents, found {df['intent'].nunique()}")
    
    print(f"Dataset: {len(df)} samples, {df['intent'].nunique()} intents")
    
    # Create mappings from actual data
    unique_intents = sorted(df['intent'].unique())
    intent_to_id = {intent: idx for idx, intent in enumerate(unique_intents)}
    id_to_intent = {idx: intent for intent, idx in intent_to_id.items()}
    num_labels = len(unique_intents)
    
    print(f"Intent mapping: {intent_to_id}")
    
    # Split data
    can_stratify = df['intent'].value_counts().min() > 1
    if can_stratify and len(df) >= num_labels * 2:
        try:
            test_size = max(0.2, min(0.4, num_labels / len(df)))
            X_train, X_test, y_train, y_test = train_test_split(
                df['query'].tolist(),
                df['intent'].tolist(),
                test_size=test_size,
                stratify=df['intent'].tolist(),
                random_state=42
            )
            print(f"Using stratified split: {len(X_train)} train, {len(X_test)} test")
        except ValueError:
            # Fallback to random split
            X_train, X_test, y_train, y_test = train_test_split(
                df['query'].tolist(),
                df['intent'].tolist(),
                test_size=0.3,
                random_state=42
            )
            print(f"Using random split: {len(X_train)} train, {len(X_test)} test")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            df['query'].tolist(),
            df['intent'].tolist(),
            test_size=0.3,
            random_state=42
        )
        print(f"Using random split: {len(X_train)} train, {len(X_test)} test")
    
    # Convert intents to IDs
    y_train_ids = [intent_to_id[intent] for intent in y_train]
    y_test_ids = [intent_to_id[intent] for intent in y_test]
    
    # Load model components
    MODEL_NAME = "bert-base-uncased"
    print(f"Loading {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id_to_intent,
        label2id=intent_to_id
    ).to(device)
    
    # Create datasets and loaders
    BATCH_SIZE = 4 if device.type == 'cpu' else 8
    MAX_LENGTH = 128
    
    train_dataset = IntentDataset(X_train, y_train_ids, tokenizer, MAX_LENGTH)
    test_dataset = IntentDataset(X_test, y_test_ids, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Training setup
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Training setup: {EPOCHS} epochs, batch size {BATCH_SIZE}, lr {LEARNING_RATE}")
    
    # Training loop
    print("Starting training...")
    model.train()
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Progress update
            if batch_idx % max(1, len(train_loader) // 5) == 0:
                batch_acc = (predictions == labels).float().mean().item()
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={batch_acc:.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        epoch_acc = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1} Summary: Loss={avg_loss:.4f}, Accuracy={epoch_acc:.4f}")
    
    print("\nTraining completed!")
    
    # Evaluation
    print("Evaluating on test set...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Classification report
    intent_names = [id_to_intent[i] for i in sorted(set(all_labels))]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=intent_names))
    
    # Confusion matrix
    if len(set(all_labels)) > 1:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=intent_names, yticklabels=intent_names)
        plt.title(f'Confusion Matrix - {accuracy*100:.1f}% Accuracy')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Confusion matrix saved as confusion_matrix.png")
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model with proper configuration
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Fix the config.json to include model_type
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Ensure model_type is set correctly
        model_config["model_type"] = "bert"
        model_config["id2label"] = id_to_intent
        model_config["label2id"] = intent_to_id
        
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
    
    # Save intent mappings
    with open(os.path.join(output_dir, "intent_mapping.json"), 'w') as f:
        json.dump(intent_to_id, f, indent=2)
    
    # Save training config
    training_config = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "num_labels": num_labels,
        "accuracy": float(accuracy),
        "intents": list(unique_intents),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    }
    
    with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
        json.dump(training_config, f, indent=2)
    
    print("Model saved successfully!")
    print(f"Final accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    return accuracy

# Test inference function
def test_inference(model_dir="./intent_model"):
    """Test the trained model with sample queries."""
    print(f"\nTesting inference...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    
    # Load intent mapping
    with open(os.path.join(model_dir, "intent_mapping.json"), 'r') as f:
        intent_to_id = json.load(f)
    
    id_to_intent = {v: k for k, v in intent_to_id.items()}
    
    # Test queries
    test_queries = [
        "Hello there, how can I get started?",
        "I need to submit a technical requirement",
        "What are your contact details?",
        "Thank you for the excellent support!",
        "I have a general question",
        "I want to share my feedback",
        "I would like to apply for a job"
    ]
    
    print("\nTest Results:")
    print("=" * 60)
    
    for query in test_queries:
        inputs = tokenizer(query, return_tensors="pt", padding=True, 
                          truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_id].item()
        
        intent = id_to_intent[predicted_id]
        print(f"'{query[:45]}...' -> {intent} ({confidence:.3f})")
    
    return True

if __name__ == "__main__":
    try:
        # Train the model
        accuracy = train_intent_classifier()
        
        # Test inference
        test_inference()
        
        print(f"\n{'='*60}")
        print("SUCCESS! Your model is trained and ready!")
        print("='*60}")
        print(f"Final accuracy: {accuracy*100:.1f}%")
        print("Model saved to: ./intent_model")
        print("Run inference.py to test with more examples")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()