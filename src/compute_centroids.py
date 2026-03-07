import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "..", "models", "embeddings.pkl")
CENTROIDS_PATH = os.path.join(BASE_DIR, "..", "models", "centroids.pkl")

with open(EMBEDDINGS_PATH, "rb") as f:
    data = pickle.load(f)

X = data["embeddings"]
y = data["labels"]
label_map = data.get("label_map", {})
person_order = data.get("person_order", None)

centroids = {}
for label in np.unique(y):
    centroids[int(label)] = X[y == label].mean(axis=0)

# ✅ نحفظ centroids + label_map معًا (بدل حفظ centroids فقط)
centroids_data = {
    "centroids": centroids,     # {0: vec, 1: vec}
    "label_map": label_map,     # {0:"RAIDAN", 1:"mohammed"} (حسب embeddings.pkl)
    "person_order": person_order
}

os.makedirs(os.path.dirname(CENTROIDS_PATH), exist_ok=True)

with open(CENTROIDS_PATH, "wb") as f:
    pickle.dump(centroids_data, f)

print("✅ Centroids + label_map saved successfully.")
print(f"Labels in centroids: {list(centroids.keys())}")
print(f"Label map: {label_map}")
