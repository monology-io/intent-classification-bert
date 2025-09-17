"""
🤖 BERT Intent Classification Training for Google Colab - Final Version
====================================================================

Instructions:
1. Upload your CSV file with 'query' and 'intent' columns
2. Choose your model (BERT or DistilBERT)
3. Watch the training progress
4. Test with inference examples for all intents

Ready? Let's train your AI model!
"""

#=============================================================================
# 📦 Install Packages & Setup
#=============================================================================
print("🔧 Installing packages...")
!pip install --upgrade transformers datasets torch scikit-learn matplotlib seaborn -q

import pandas as pd
import numpy as np
import torch
import os
from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Setup complete! Using: {device}")

#=============================================================================
# 📊 Load Dataset
#=============================================================================
print("\n📊 Upload your CSV file...")
from google.colab import files
uploaded = files.upload()

if not uploaded:
    print("❌ No file uploaded.")
    exit()

file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)

# Validate and clean data
if 'query' not in df.columns or 'intent' not in df.columns:
    print("❌ CSV must have 'query' and 'intent' columns.")
    exit()

df.dropna(subset=['query', 'intent'], inplace=True)
df['intent'] = df['intent'].astype(str)
df['query'] = df['query'].astype(str)
df.reset_index(drop=True, inplace=True)

if df['intent'].nunique() < 2:
    print(f"❌ Need at least 2 intents, found {df['intent'].nunique()}")
    exit()

print(f"✅ Loaded {len(df)} samples with {df['intent'].nunique()} intents")

#=============================================================================
# ⚙️ Configuration
#=============================================================================
print("\nChoose model: 1) BERT (accurate) 2) DistilBERT (faster)")
model_choice = input("Enter 1 or 2 [default: 1]: ") or "1"
MODEL_NAME = "bert-base-uncased" if model_choice == "1" else "distilbert-base-uncased"

MAX_LENGTH = 128
BATCH_SIZE = 8 if torch.cuda.is_available() else 4
EPOCHS = 3
LEARNING_RATE = 2e-5

print(f"🎯 Training: {MODEL_NAME}, {EPOCHS} epochs, batch size {BATCH_SIZE}")

#=============================================================================
# 🔧 Data Preprocessing
#=============================================================================
print("\n🔧 Preprocessing...")

# Create mappings
unique_intents = sorted(df['intent'].unique())
intent_mapping = {intent: i for i, intent in enumerate(unique_intents)}
id_to_intent = {i: intent for intent, i in intent_mapping.items()}
num_labels = len(unique_intents)

# Convert to Dataset with ClassLabel
dataset = Dataset.from_pandas(df)
cl = ClassLabel(num_classes=num_labels, names=unique_intents)
dataset = dataset.cast_column("intent", cl)

# Smart data splitting
can_stratify = df['intent'].value_counts().min() > 1
if can_stratify:
    test_ratio = max(0.2, min(0.4, num_labels / len(df)))
    dataset_split = dataset.train_test_split(test_size=test_ratio, stratify_by_column="intent")
    print(f"✅ Stratified split: {len(dataset_split['train'])} train, {len(dataset_split['test'])} test")
else:
    dataset_split = dataset.train_test_split(test_size=0.3)
    print(f"⚠️ Random split: {len(dataset_split['train'])} train, {len(dataset_split['test'])} test")

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    tokenized = tokenizer(examples["query"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    tokenized["labels"] = examples["intent"]
    return tokenized

encoded_dataset = dataset_split.map(preprocess_function, batched=True)

#=============================================================================
# 🤖 Model Setup & Training
#=============================================================================
print(f"\n🤖 Loading {MODEL_NAME}...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id_to_intent,
    label2id=intent_mapping,
    problem_type="single_label_classification"
).to(device)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=50,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("🚀 Training started...")
trainer.train()

#=============================================================================
# 📊 Evaluation
#=============================================================================
print("\n📊 Evaluating...")
eval_results = trainer.evaluate()
accuracy = eval_results['eval_accuracy']
print(f"🎯 Final Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

# Quick confusion matrix visualization
predictions = trainer.predict(encoded_dataset["test"])
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_labels, pred_labels)
target_names = [model.config.id2label[i] for i in sorted(set(true_labels))]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title(f'Confusion Matrix - {accuracy*100:.1f}% Accuracy')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#=============================================================================
# 💾 Save Model
#=============================================================================
model_path = "./intent_model"
print(f"\n💾 Saving model to {model_path}...")
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
print("✅ Model saved!")

#=============================================================================
# 🧪 Inference Testing for All Intents
#=============================================================================
def predict_intent(query):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        predicted_id = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_id].item()
    return model.config.id2label[predicted_id], confidence

print(f"\n🧪 Testing inference for all {num_labels} intents:")
print("=" * 70)

# Create sample queries for each intent based on training data
intent_examples = {}
for intent in unique_intents:
    # Get a few examples from training data for each intent
    examples = df[df['intent'] == intent]['query'].head(3).tolist()
    intent_examples[intent] = examples

# Test each intent with examples
for intent, examples in intent_examples.items():
    print(f"\n🏷️ Intent: {intent}")
    print("-" * 40)
    
    for query in examples[:2]:  # Test 2 examples per intent
        predicted_intent, confidence = predict_intent(query)
        status = "✅" if predicted_intent == intent else "❌"
        print(f"{status} '{query[:50]}...' → {predicted_intent} ({confidence:.2%})")

#=============================================================================
# 🎉 Interactive Testing
#=============================================================================
print(f"\n" + "=" * 70)
print("🎉 MODEL TRAINING COMPLETE!")
print("=" * 70)
print(f"📊 Dataset: {len(df)} samples, {num_labels} intents")
print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"💾 Model saved to: {model_path}")
print(f"🔧 Ready for experimental use!")

print(f"\n🔬 Test your own queries:")
while True:
    user_query = input("\nEnter query (or 'quit'): ")
    if user_query.lower() in ['quit', 'exit', 'q']:
        break
    if user_query.strip():
        intent, confidence = predict_intent(user_query)
        print(f"🎯 {intent} ({confidence:.1%})")

print("\n✨ Happy coding! Your AI model is ready! 🚀")