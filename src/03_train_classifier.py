import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# =========================
# PATH CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "..", "models", "embeddings.pkl")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "..", "models", "svm_classifier.pkl")

# =========================
# LOAD EMBEDDINGS
# =========================
with open(EMBEDDINGS_PATH, "rb") as f:
    data = pickle.load(f)

X = data["embeddings"]
y = data["labels"]
label_map = data["label_map"]

print("Embeddings shape:", X.shape)
print("Labels shape:", y.shape)

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# TRAIN CLASSIFIER
# =========================
classifier = SVC(kernel="linear", probability=True)
classifier.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_map.values()))

# =========================
# SAVE CLASSIFIER
# =========================
with open(CLASSIFIER_PATH, "wb") as f:
    pickle.dump(classifier, f)

print("✅ Classifier saved successfully.")
