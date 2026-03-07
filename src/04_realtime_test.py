import os
import cv2
import torch
import pickle
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
DISTANCE_THRESHOLD = 0.75  # يمكن ضبطها لاحقًا

# =========================
# PATH CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CENTROIDS_PATH = os.path.join(BASE_DIR, "..", "models", "centroids.pkl")

# =========================
# LOAD CENTROIDS (+ label_map if available)
# =========================
with open(CENTROIDS_PATH, "rb") as f:
    loaded = pickle.load(f)

# يدعم الشكل الجديد: {"centroids":..., "label_map":...}
# ويدعم الشكل القديم: centroids فقط
if isinstance(loaded, dict) and "centroids" in loaded:
    centroids = loaded["centroids"]
    label_map = loaded.get("label_map", {})
else:
    centroids = loaded
    label_map = {}

# إذا label_map فيه person_1 / person_2 نحولها لأسماء العرض المطلوبة
DISPLAY_NAME_MAP = {
    "person_1": "RAIDAN",
    "person_2": "mohammed",
}

def get_display_name(label_id):
    # label_id غالبًا رقم: 0 أو 1
    raw = label_map.get(int(label_id), str(label_id)) if isinstance(label_id, (int, np.integer)) else str(label_id)
    return DISPLAY_NAME_MAP.get(raw, raw)

# =========================
# INITIALIZE MODELS
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# =========================
# OPEN CAMERA
# =========================
cap = cv2.VideoCapture(0)
print("🎥 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Detect faces + boxes
    boxes, probs = mtcnn.detect(img_pil)

    if boxes is not None and len(boxes) > 0:
        faces = mtcnn.extract(img_pil, boxes, save_path=None)

        if faces is not None:
            # ✅ إصلاح شكل الـTensor:
            # أحيانًا يرجع (3,160,160) لوجه واحد، لازم نخليه (1,3,160,160)
            if isinstance(faces, list):
                faces = torch.stack(faces, dim=0)

            if isinstance(faces, torch.Tensor) and faces.dim() == 3:
                faces = faces.unsqueeze(0)

            faces = faces.to(device)

            with torch.no_grad():
                embeddings = facenet(faces).cpu().numpy()  # (N, 512)

            # ارسم لكل وجه
            for box, emb in zip(boxes, embeddings):
                x1, y1, x2, y2 = [int(v) for v in box]

                similarities = {
                    label: cosine_similarity(
                        emb.reshape(1, -1),
                        centroid.reshape(1, -1)
                    )[0][0]
                    for label, centroid in centroids.items()
                }

                best_label = max(similarities, key=similarities.get)
                best_similarity = similarities[best_label]

                if best_similarity < DISTANCE_THRESHOLD:
                    name = "Unknown"
                    color = (0, 0, 255)
                else:
                    name = get_display_name(best_label)  # ✅ RAIDAN / mohammed
                    color = (0, 255, 0)

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Name above box
                text = name
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.8
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

                tx, ty = x1, y1 - 10
                if ty - th < 0:
                    ty = y1 + th + 10

                cv2.rectangle(frame, (tx, ty - th - baseline), (tx + tw, ty + baseline), color, -1)
                cv2.putText(frame, text, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
