import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.svm import SVC


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

vectorizer = TfidfVectorizer()

X_train_log = vectorizer.fit_transform(train_data["Further Comments"])
y_train_log = train_data["Type"]

X_val_log = vectorizer.transform(val_data["Further Comments"])
y_val_log = val_data["Type"]

X_test_log = vectorizer.transform(test_data["Further Comments"])
y_test_log = test_data["Type"]

# Temporarily map 'Type' column to class names for display
temp_class_names = test_data['Type'].map(class_mapping)

# Display value counts with class names
print(temp_class_names.value_counts())


"""Decision Tree Classifier - tfidf"""

# Step 1: Train the Decision Tree model
dt_model = DecisionTreeClassifier(
    criterion='entropy',  # Splitting criterion ('gini' or 'entropy')
    max_depth=None,    # Maximum depth of the tree (None means unlimited depth)
    random_state=42    # Ensures reproducibility
)
dt_model.fit(X_train_log, y_train_log)

# Step 2: Evaluate the model on the test dataset
y_pred = dt_model.predict(X_test_log)

# Step 3: Predict probabilities for AUC calculation
# Note: DecisionTreeClassifier provides `predict_proba` by default
y_pred_prob = dt_model.predict_proba(X_test_log)

# Step 4: Calculate accuracy
accuracy = accuracy_score(y_test_log, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 5: Print classification report
print("\nClassification Report:")
print(classification_report(y_test_log, y_pred, target_names = target_names))

# Step 6: Calculate AUC score
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test_log)  # Binarize true labels
auc_score = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr')
print(f"AUC Score: {auc_score:.4f}")


"""Logidtic Regression - tfidf"""

base_model = LogisticRegression(max_iter=1000, random_state=42)
base_model.fit(X_train_log, y_train_log)
y_pred = base_model.predict(X_test_log)
y_pred_prob = base_model.predict_proba(X_test_log)

# Calculate accuracy
accuracy = accuracy_score(y_test_log, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test_log, y_pred, target_names = target_names))

# Calculate AUC score
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test_log)  # Binarize true labels
auc_score = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr')
print(f"AUC Score: {auc_score:.4f}")


"""Random forest - tfidf"""

rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)  # 200 fully grown trees
rf_model.fit(X_train_log, y_train_log)

# Step 3: Evaluate the model on the test dataset
y_pred = rf_model.predict(X_test_log)

y_pred_prob = rf_model.predict_proba(X_test_log)

# Calculate accuracy
accuracy = accuracy_score(y_test_log, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test_log, y_pred, target_names = target_names))

# Calculate AUC score
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test_log)  # Binarize true labels
auc_score = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr')
print(f"AUC Score: {auc_score:.4f}")

"""SVC - tfidf"""

# Step 1: Train the SVC model with probability=True
svm_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)  # Enable probability=True
svm_model.fit(X_train_log, y_train_log)

# Step 2: Evaluate the model on the test dataset
y_pred = svm_model.predict(X_test_log)

# Step 3: Predict probabilities for AUC calculation
y_pred_prob = svm_model.predict_proba(X_test_log)

# Step 4: Calculate accuracy
accuracy = accuracy_score(y_test_log, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 5: Print classification report
print("\nClassification Report:")
print(classification_report(y_test_log, y_pred, target_names = target_names))

# Step 6: Calculate AUC score
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test_log)  # Binarize true labels
auc_score = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr')
print(f"AUC Score: {auc_score:.4f}")

"""XGBoost - tfidf"""

# Step 1: Train the XGBoost model
xgb_model = XGBClassifier(
    objective='multi:softprob',  # Multi-class classification with probability outputs
    num_class=len(y_train_log.unique()),  # Number of unique classes
    eval_metric='mlogloss',  # Multi-class log loss
    use_label_encoder=False,  # Avoid warnings in XGBoost
    random_state=42
)
xgb_model.fit(X_train_log, y_train_log)

# Step 2: Evaluate the model on the test dataset
y_pred = xgb_model.predict(X_test_log)

# Step 3: Predict probabilities for AUC calculation
y_pred_prob = xgb_model.predict_proba(X_test_log)

# Step 4: Calculate accuracy
accuracy = accuracy_score(y_test_log, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 5: Print classification report
print("\nClassification Report:")
print(classification_report(y_test_log, y_pred, target_names=target_names))

# Step 6: Calculate AUC score
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test_log)  # Binarize true labels
auc_score = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr')
print(f"AUC Score: {auc_score:.4f}")

"""LightGBM - tfidf"""

lgb_train = lgb.Dataset(X_train_log, label=y_train_log)
lgb_val = lgb.Dataset(X_val_log, label=y_val_log, reference=lgb_train)

params = {
    'objective': 'multiclass',  # Multi-class classification
    'num_class': len(y_train_log.unique()),  # Number of classes
    'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
    'metric': 'multi_logloss',  # Loss function
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9
}

lgbm_model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],
    num_boost_round=100,
)

# Step 3: Make predictions on the test dataset
y_pred_prob = lgbm_model.predict(X_test_log)  # Predict probabilities for all classes
y_pred = y_pred_prob.argmax(axis=1)  # Convert probabilities to class predictions

# Step 4: Evaluate the model
# Calculate accuracy
accuracy = accuracy_score(y_test_log, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test_log, y_pred, target_names=target_names))

# Calculate AUC score (One-vs-Rest for multi-class)
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test_log)  # Binarize true labels
auc_score = roc_auc_score(y_test_bin, y_pred_prob, multi_class='ovr')
print(f"AUC Score: {auc_score:.4f}")

"""Graph - tfidf"""

import numpy as np
import matplotlib.pyplot as plt

# Model metrics from the provided results
metrics = {
    "Logistic Regression": [0.80, 0.81, 0.77, 0.8109, 0.9628],
    "Decision Tree": [0.85, 0.86, 0.85, 0.8574, 0.8653],
    "Random Forest": [0.91, 0.90, 0.89, 0.8959, 0.9754],
    "SVC": [0.86, 0.86, 0.83, 0.8567, 0.9771],
    "XGBoost": [0.87, 0.87, 0.85, 0.8656, 0.9647],
    "LightGBM": [0.90, 0.90, 0.90, 0.9026, 0.9709],
}

# Metrics labels
metric_labels = ["Precision", "Recall", "F1-Score", "Accuracy", "AUC"]

# Convert to array for easier plotting
model_names = list(metrics.keys())
values = np.array(list(metrics.values()))

# Custom color palette
colors = ["#8e44ad", "#1f6edb", "#00bce8", "#8bc34a", "#fdd835", "#f6a623", "#d64d12"]

# Create bar graph
bar_width = 0.15
x = np.arange(len(metric_labels))  # X locations for the metrics

plt.figure(figsize=(10, 6))

for i, (model, color) in enumerate(zip(model_names, colors)):
    plt.bar(x + i * bar_width, values[i], width=bar_width, label=model, color=color)

# Formatting the plot
plt.xlabel("Metrics")
plt.ylabel("Values")
plt.title("ML Model (with TF-IDF) Comparison")
plt.xticks(x + bar_width * (len(model_names) - 1) / 2, metric_labels)
plt.ylim(0.7, 1.0)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()