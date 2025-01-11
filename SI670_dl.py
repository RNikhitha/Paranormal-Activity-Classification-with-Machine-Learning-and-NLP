import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from autocorrect import Speller
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from transformers import AlbertTokenizer, AlbertForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification
import torch
import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv("dataset.csv")
df.head()

duplicate_rows = df[df.duplicated()]
print("Duplicate Rows:")
print(duplicate_rows)

df = df.drop_duplicates()

df.to_csv('unique_dataset.csv', index=False)

unique_data = pd.read_csv("unique_dataset.csv")
unique_data.shape

# Remove the column "Date/Time" as it is irrelevant
unique_data = unique_data.drop(columns=['Date / Time'], axis=1)
unique_data.head()

type_mapping = {
    'Haunting Manifestation': 'Haunting Manifestation',
    'Post-Mortem Manifestation': 'Post-Mortem Manifestation',
    'Crisis Manifestation': 'Crisis Manifestation',
    'Manifestation of the Living': 'Environmental Manifestation',
    'Legend': 'Legend',
    'Legend - Old Nick': 'Legend',
    'legend': 'Legend',
    'Fairy': 'Fairy',
    'Shuck': 'Mythical',
    'Other': 'Other',
    'Cryptozoology': 'Cryptozoology',
    "Poltergeist": "Poltergeist",
    'ABC': 'ABC',
    'UFO': 'UFO',
    'Dragon': 'Mythical',
    'Environmental Manifestation': 'Environmental Manifestation',
    'Curse': 'Curse',
    'Vampire': 'Mythical',
    'Werewolf': 'Mythical',
    'SHC': 'Other',
    'Experimental Manifestation': 'Other',
    'Unknown Ghost Type': 'Unknown Ghost Type'
}

# Replace the Type column with the mapped values
unique_data['Type'] = unique_data['Type'].map(type_mapping)

# Group by the new type and sum counts
combined_df = unique_data["Type"].value_counts()

# Print the new class distribution
combined_df

# Printing unqiue categories
categories = unique_data['Type'].unique()
num = len(categories)
print(f"Number of categories: {num}")
print(categories)

# Label encoding
label_encoder = LabelEncoder()
unique_data['Type'] = label_encoder.fit_transform(unique_data['Type'])

# Print the class-to-number mapping
class_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
print(class_mapping)

target_names = [class_mapping[i] for i in sorted(class_mapping.keys())]

# Converting the text into lowercase
unique_data['Title'] = unique_data['Title'].str.lower()
unique_data['Location'] = unique_data['Location'].str.lower()
unique_data['Further Comments'] = unique_data['Further Comments'].str.lower()

# Removing punctuations
unique_data['Title'] = unique_data['Title'].str.replace('[^\w\s]', '', regex=True)
unique_data['Location'] = unique_data['Location'].str.replace('[^\w\s]', '', regex=True)
unique_data['Further Comments'] = unique_data['Further Comments'].str.replace('[^\w\s]', '', regex=True)

nltk.download('stopwords')
stop = set(stopwords.words('english'))

unique_data['Title'] = unique_data['Title'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
unique_data['Location'] = unique_data['Location'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
unique_data['Further Comments'] = unique_data['Further Comments'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

# Step 5: Remove special characters
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

unique_data['Title'] = unique_data['Title'].apply(clean_text)
unique_data['Further Comments'] = unique_data['Further Comments'].apply(clean_text)

# Step 9: Handle contractions (e.g., don't -> do not)
contractions = {"don't": "do not", "can't": "cannot", "isn't": "is not", "i'm": "i am", "it's": "it is"}
def expand_contractions(text):
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)
    return text

unique_data['Title'] = unique_data['Title'].apply(expand_contractions)
unique_data['Further Comments'] = unique_data['Further Comments'].apply(expand_contractions)

metrics_data = { 'Model': [], 'Class': [], 'Metric': [], 'Score': []}
metrics_data = pd.DataFrame(metrics_data)
metrics_data.head()

train_data, test_data = train_test_split(unique_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

"""BERT"""

# Create HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_data[['Further Comments', 'Type']])
test_dataset = Dataset.from_pandas(test_data[['Further Comments', 'Type']])
val_dataset = Dataset.from_pandas(val_data[['Further Comments', 'Type']])

# Load the XLNet tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# Tokenization function
def tokenize_data(example):
    return tokenizer(example['Further Comments'], padding='max_length', truncation=True)

def transform_labels(example):
    type_label = int(example['Type'])  # Convert 'Type' to integer if it's not already
    return {'labels': type_label}


def preprocess_dataset(dataset):
    # Tokenize the dataset
    dataset = dataset.map(tokenize_data, batched=True)
    dataset = dataset.map(transform_labels)
    return dataset

train_dataset = preprocess_dataset(Dataset.from_pandas(train_data[['Further Comments', 'Type']]))
val_dataset = preprocess_dataset(Dataset.from_pandas(val_data[['Further Comments', 'Type']]))
test_dataset = preprocess_dataset(Dataset.from_pandas(test_data[['Further Comments', 'Type']]))

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load XLNet model for sequence classification
num_labels = unique_data['Type'].nunique()  # Number of unique types
model =  AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=14)

# Define training arguments
training_args = TrainingArguments("test_trainer", num_train_epochs=7, report_to="none", per_device_train_batch_size=32, per_device_eval_batch_size=32)

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)  # Convert logits to class predictions
    acc = accuracy_score(labels, preds)  # Accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracAy": acc, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate(test_dataset)
print("Test Results:", results)

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import numpy as np

# Function to generate predictions, calculate accuracy, and print the classification report
def evaluate_classification_report(trainer, dataset, label_names):
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=1)  # Convert logits to class predictions
    labels = predictions.label_ids  # True labels

    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

    # Generate and print the classification report
    report = classification_report(labels, preds, target_names=label_names, zero_division=1)
    print("Classification Report:\n", report)

    auc = roc_auc_score(labels, predictions.predictions, multi_class='ovr', average='weighted')
    print(f"AUC (Area Under the Curve): {auc:.4f}")

# Get unique class names for the report
label_names = [str(label) for label in sorted(train_data['Type'].unique())]

# Print classification report and accuracy for the test dataset
evaluate_classification_report(trainer, test_dataset, label_names)

# Save the trained model
model_save_path = "/content/bert_model"
model.save_pretrained(model_save_path)  # Save model and configuration
tokenizer.save_pretrained(model_save_path)  # Save tokenizer

print(f"Model and tokenizer saved to {model_save_path}")


"""XLNet"""

# Create HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_data[['Further Comments', 'Type']])
test_dataset = Dataset.from_pandas(test_data[['Further Comments', 'Type']])
val_dataset = Dataset.from_pandas(val_data[['Further Comments', 'Type']])

# Load the XLNet tokenizer
tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')

# Tokenization function
def tokenize_data(example):
    return tokenizer(example['Further Comments'], padding='max_length', truncation=True, max_length=128)

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_data, batched=True)
test_dataset = test_dataset.map(tokenize_data, batched=True)
val_dataset = val_dataset.map(tokenize_data, batched=True)

# Set the format for PyTorch
train_dataset = train_dataset.rename_column("Type", "labels")
test_dataset = test_dataset.rename_column("Type", "labels")
val_dataset = val_dataset.rename_column("Type", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load XLNet model for sequence classification
num_labels = unique_data['Type'].nunique()  # Number of unique types
model = AutoModelForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=num_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./xlnet_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

# Metrics function
def compute_metrics(pred):
    predictions, labels = pred
    predictions = predictions.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate(test_dataset)
print("Test Results:", results)

# Create a new Trainer with the loaded model
test_trainer = Trainer(model=saved_model)

# Predict on the test dataset
predictions = test_trainer.predict(test_dataset)

# Access predictions and labels
predicted_labels = predictions.predictions.argmax(axis=-1)
true_labels = predictions.label_ids

# Evaluate predictions
from sklearn.metrics import classification_report
print(classification_report(true_labels, predicted_labels))


"""ALBERT and ELECTRA"""

# Split the dataset into train, validation, and test
train_data, test_data = train_test_split(unique_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(train_data["Type"])
y_val_encoded = label_encoder.transform(val_data["Type"])
y_test_encoded = label_encoder.transform(test_data["Type"])

# Tokenize the data using ALBERT and ELECTRA Tokenizers
albert_tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
electra_tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")

train_data["Further Comments"] = train_data["Further Comments"].astype(str)
val_data["Further Comments"] = val_data["Further Comments"].astype(str)
test_data["Further Comments"] = test_data["Further Comments"].astype(str)

# Define Dataset class for tokenization and model input
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
# Prepare datasets
train_dataset_albert = TextDataset(train_data["Further Comments"].tolist(), y_train_encoded, albert_tokenizer)
val_dataset_albert = TextDataset(val_data["Further Comments"].tolist(), y_val_encoded, albert_tokenizer)
test_dataset_albert = TextDataset(test_data["Further Comments"].tolist(), y_test_encoded, albert_tokenizer)

train_dataset_electra = TextDataset(train_data["Further Comments"].tolist(), y_train_encoded, electra_tokenizer)
val_dataset_electra = TextDataset(val_data["Further Comments"].tolist(), y_val_encoded, electra_tokenizer)
test_dataset_electra = TextDataset(test_data["Further Comments"].tolist(), y_test_encoded, electra_tokenizer)

# Create DataLoaders
train_loader_albert = DataLoader(train_dataset_albert, batch_size=8, shuffle=True)
val_loader_albert = DataLoader(val_dataset_albert, batch_size=8, shuffle=False)
test_loader_albert = DataLoader(test_dataset_albert, batch_size=8, shuffle=False)

train_loader_electra = DataLoader(train_dataset_electra, batch_size=8, shuffle=True)
val_loader_electra = DataLoader(val_dataset_electra, batch_size=8, shuffle=False)
test_loader_electra = DataLoader(test_dataset_electra, batch_size=8, shuffle=False)

# Load models
albert_model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=len(label_encoder.classes_))
electra_model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=len(label_encoder.classes_))

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
albert_model.to(device)
electra_model.to(device)

# Define optimizer
optimizer_albert = AdamW(albert_model.parameters(), lr=1e-5)
optimizer_electra = AdamW(electra_model.parameters(), lr=1e-5)

import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Define the Softmax layer to convert logits into probabilities
softmax = nn.Softmax(dim=1)

# Update training and evaluation function to handle probabilities for AUC
def train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, device):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Evaluation on validation set
    model.eval()
    y_true_val = []
    y_pred_val = []
    y_prob_val = []  # To store the probabilities for AUC
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            probabilities = softmax(logits)  # Get the probabilities

            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(predictions.cpu().numpy())
            y_prob_val.extend(probabilities.cpu().numpy())  # Store the probabilities

    # Calculate metrics on validation data
    accuracy_val = accuracy_score(y_true_val, y_pred_val)
    precision_val = precision_score(y_true_val, y_pred_val, average='weighted', zero_division=1)
    recall_val = recall_score(y_true_val, y_pred_val, average='weighted', zero_division=1)
    f1_val = f1_score(y_true_val, y_pred_val, average='weighted', zero_division=1)

    # For multi-class AUC, we need to pass probabilities for each class
    auc_val = roc_auc_score(y_true_val, np.array(y_prob_val), multi_class='ovr', average='weighted')

    # Final evaluation on test set
    y_true_test = []
    y_pred_test = []
    y_prob_test = []  # To store the probabilities for AUC
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            probabilities = softmax(logits)  # Get the probabilities

            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(predictions.cpu().numpy())
            y_prob_test.extend(probabilities.cpu().numpy())  # Store the probabilities

    # Calculate metrics on test data
    accuracy_test = accuracy_score(y_true_test, y_pred_test)
    precision_test = precision_score(y_true_test, y_pred_test, average='weighted', zero_division=1)
    recall_test = recall_score(y_true_test, y_pred_test, average='weighted', zero_division=1)
    f1_test = f1_score(y_true_test, y_pred_test, average='weighted', zero_division=1)

    # For multi-class AUC, we need to pass probabilities for each class
    auc_test = roc_auc_score(y_true_test, np.array(y_prob_test), multi_class='ovr', average='weighted')

    return (accuracy_val, precision_val, recall_val, f1_val, auc_val), (accuracy_test, precision_test, recall_test, f1_test, auc_test)

# Train and evaluate ALBERT
val_metrics_albert, test_metrics_albert = train_and_evaluate(albert_model, train_loader_albert, val_loader_albert, test_loader_albert, optimizer_albert, device)

print("ALBERT Validation Results:")
print(f"Accuracy: {val_metrics_albert[0]:.4f}")
print(f"Precision: {val_metrics_albert[1]:.4f}")
print(f"Recall: {val_metrics_albert[2]:.4f}")
print(f"F1 Score: {val_metrics_albert[3]:.4f}")
print(f"AUC: {val_metrics_albert[4]:.4f}")

print("ALBERT Test Results:")
print(f"Accuracy: {test_metrics_albert[0]:.4f}")
print(f"Precision: {test_metrics_albert[1]:.4f}")
print(f"Recall: {test_metrics_albert[2]:.4f}")
print(f"F1 Score: {test_metrics_albert[3]:.4f}")
print(f"AUC: {test_metrics_albert[4]:.4f}")

# Train and evaluate ELECTRA
val_metrics_electra, test_metrics_electra = train_and_evaluate(electra_model, train_loader_electra, val_loader_electra, test_loader_electra, optimizer_electra, device)

print("\nELECTRA Validation Results:")
print(f"Accuracy: {val_metrics_electra[0]:.4f}")
print(f"Precision: {val_metrics_electra[1]:.4f}")
print(f"Recall: {val_metrics_electra[2]:.4f}")
print(f"F1 Score: {val_metrics_electra[3]:.4f}")
print(f"AUC: {val_metrics_electra[4]:.4f}")

print("ELECTRA Test Results:")
print(f"Accuracy: {test_metrics_electra[0]:.4f}")
print(f"Precision: {test_metrics_electra[1]:.4f}")
print(f"Recall: {test_metrics_electra[2]:.4f}")
print(f"F1 Score: {test_metrics_electra[3]:.4f}")
print(f"AUC: {test_metrics_electra[4]:.4f}")

"""Neural Attention Forest"""

# Split data into train, validation, and test first
X_train, X_temp, y_train, y_temp = train_test_split(df['Further Comments'], df['Type'], test_size=0.2, random_state=42)

# Split X_temp and y_temp into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit the vectorizer on the training data and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()

# Apply the same transformation to validation and test data (using the already fitted vectorizer)
X_val_tfidf = tfidf_vectorizer.transform(X_val).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

# Initialize and fit LabelEncoder on the training labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Apply the same transformation to validation and test labels
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

class AttentionLayer(layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.dense = layers.Dense(units, activation='tanh')
        self.attention_weights = layers.Dense(units, activation='softmax')  # Apply softmax across features

    def call(self, inputs):
        dense_output = self.dense(inputs)  # (batch_size, features, units)
        weights = self.attention_weights(dense_output)  # (batch_size, features, units)
        weighted_inputs = inputs * weights  # Element-wise multiplication
        return tf.reduce_sum(weighted_inputs, axis=1)  # Summing across the feature axis


def build_neural_attention_forest(hp, input_dim, num_classes):
    # Define the model
    inputs = layers.Input(shape=(input_dim,))
    outputs = []
    num_trees = hp.Int('num_trees', min_value=3, max_value=10, step=1)
    tree_depth = hp.Int('tree_depth', min_value=2, max_value=6, step=1)
    units = hp.Int('units', min_value=32, max_value=128, step=32)
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')

    for _ in range(num_trees):
        x = inputs
        for _ in range(tree_depth):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
        x = layers.Reshape((-1, 1))(x)  # Reshape for compatibility with Attention Layer
        x = AttentionLayer(units=units)(x)
        outputs.append(x)

    # Combine tree outputs
    combined = layers.concatenate(outputs)
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dense(num_classes, activation='softmax')(combined)

    # Compile the model
    model = models.Model(inputs=inputs, outputs=combined)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def tune_hyperparameters(X_train, y_train, X_val, y_val, input_dim, num_classes):
    tuner = kt.RandomSearch(
        lambda hp: build_neural_attention_forest(hp, input_dim, num_classes),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='my_dir',
        project_name='neural_attention_forest'
    )

    # Perform the search
    tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

    # Get the best model after tuning
    best_model = tuner.get_best_models(num_models=1)[0]

    # Get the best trial
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]

    # Print the best hyperparameters
    best_hyperparameters = best_trial.hyperparameters.values  # Get hyperparameter values
    print("Best Hyperparameters: ", best_hyperparameters)  # Print best hyperparameters

    return best_model

# Input dimensions and classes
input_dim = X_train_tfidf.shape[1]
num_classes = len(set(y_train_encoded))

# Hyperparameter tuning
best_model = tune_hyperparameters(X_train_tfidf, y_train_encoded, X_val_tfidf, y_val_encoded, input_dim, num_classes)

# Train the best model
best_model.fit(
    X_train_tfidf, y_train_encoded,
    epochs=20,
    validation_data=(X_val_tfidf, y_val_encoded),
    batch_size=32,
    verbose=1
)

# Predictions
y_pred_prob = best_model.predict(X_test_tfidf)
y_pred = y_pred_prob.argmax(axis=1)

# Evaluation Metrics
print(classification_report(y_test_encoded, y_pred))
print(f"Accuracy: {accuracy_score(y_test_encoded, y_pred):.4f}")
print(f"AUC Score: {roc_auc_score(y_test_encoded, y_pred_prob, multi_class='ovr'):.4f}")

#save the model
best_model.save('neural_attention_bag_model.h5')


"""Graph"""

# Define the models and their metrics
models = ['BERT', 'ALBERT', 'ELECTRA', 'XLNet', 'Neural Attention Forest']
metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC']

# Metric values for each model (from the provided data)
data = [
    [0.91, 0.91, 0.91, 91.54, 97.65],  # BERT
    [0.8331, 0.8405, 0.8117, 84.05, 95.12],  # ALBERT
    [0.7732, 0.6720, 0.5761, 67.20, 84.26],  # ELECTRA
    [0.89, 0.90, 0.89, 89.52, 97.02],  # XLNet
    [0.91, 0.91, 0.91, 90.95, 96.02]   # Neural Attention Forest
]

# Convert percentages to decimals where applicable (Accuracy and AUC)
data = np.array(data) / [1, 1, 1, 100, 100]

# Custom color palette
colors = ["#8e44ad", "#1f6edb", "#00bce8", "#8bc34a", "#fdd835", "#f6a623", "#d64d12"]

# Create the bar graph
bar_width = 0.15
x = np.arange(len(metrics))

plt.figure(figsize=(10, 6))

# Add bars for each model
for i, (model, color) in enumerate(zip(models, colors)):
    plt.bar(x + i * bar_width, data[i], width=bar_width, label=model, color=color)

# Add labels, title, and legend
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.title('Model Metrics Comparison', fontsize=14)
plt.xticks(x + (len(models) - 1) * bar_width / 2, metrics)
plt.ylim(0.5, 1.0)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

"""Word Cloud"""

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/content/dataset_final.csv')

# Combine all the text in the 'Further Comments' column into one large string
text = ' '.join(df['Further Comments'].dropna().astype(str))

# Generate the word cloud
wordcloud = WordCloud(width=400, height=600, background_color='white',prefer_horizontal=1).generate(text)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.savefig('word_cloud.png')
plt.axis('off')  # No axes for the word cloud
plt.show()