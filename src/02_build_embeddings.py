import os
import cv2
import numpy as np
import torch
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1

# =========================
# PATH CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "models", "embeddings.pkl")

# =========================
# INITIALIZE MODELS
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(image_size=160, margin=20, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# =========================
# IMPORTANT: FIXED LABEL ORDER
# =========================
# هنا نحدد أسماء الأشخاص "الحقيقية" ونثبت ترتيبهم (0 ثم 1)
# لازم تكون أسماء المجلدات داخل dataset مطابقة لهذه الأسماء بالضبط:
# dataset/RAIDAN
# dataset/mohammed
PERSON_ORDER = ["RAIDAN", "mohammed"]

embeddings = []
labels = []
label_map = {}

# =========================
# PROCESS DATASET
# =========================
for label_id, person_name in enumerate(PERSON_ORDER):
    person_path = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_path):
        print(f"⚠️ Folder not found: {person_path}")
        continue

    label_map[label_id] = person_name
    print(f"Processing {person_name} as label {label_id}...")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(img_rgb)

        if face is None:
            continue

        with torch.no_grad():
            embedding = facenet(face.unsqueeze(0).to(device))

        embeddings.append(embedding.cpu().numpy()[0])
        labels.append(label_id)

# =========================
# SAVE EMBEDDINGS
# =========================
data = {
    "embeddings": np.array(embeddings),
    "labels": np.array(labels),
    "label_map": label_map,
    "person_order": PERSON_ORDER,  # اختياري للتوثيق
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(data, f)

print("✅ Face embeddings saved successfully.")
print(f"Total samples: {len(embeddings)}")
print(f"Label map: {label_map}")
