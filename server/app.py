import os
import io
import uuid
import pickle
import numpy as np
from PIL import Image

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Flask App
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

EMBEDDINGS_PATH = os.path.join(PROJECT_DIR, "models", "embeddings.pkl")
CENTROIDS_PATH = os.path.join(PROJECT_DIR, "models", "centroids.pkl")

# =========================
# CONFIG
# =========================
DISTANCE_THRESHOLD = 0.80

# =========================
# MODELS
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# =========================
# GLOBALS (in-memory)
# =========================
centroids = {}
label_map = {}

# training buffer in RAM:
# sessions[session_id] = {"name": str, "embeddings": [np.array(512), ...]}
sessions = {}

# =========================
# HELPERS
# =========================
def load_centroids():
    global centroids, label_map
    with open(CENTROIDS_PATH, "rb") as f:
        loaded = pickle.load(f)

    if isinstance(loaded, dict) and "centroids" in loaded:
        centroids = loaded["centroids"]
        label_map = loaded.get("label_map", {})
    else:
        centroids = loaded
        label_map = {}

def get_display_name(label_id):
    return label_map.get(int(label_id), str(label_id))

def read_image_from_request():
    if "image" not in request.files:
        return None
    image_file = request.files["image"]
    image_bytes = image_file.read()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def compute_embedding_from_image(img_rgb_pil):
    """
    Returns (embedding_np, box) or (None, None)
    box = [x1, y1, x2, y2] in original image coordinates
    """
    boxes, probs = mtcnn.detect(img_rgb_pil)
    if boxes is None or len(boxes) == 0:
        return None, None

    # First face only
    x1, y1, x2, y2 = map(int, boxes[0])
    face_crop = img_rgb_pil.crop((x1, y1, x2, y2))

    face = mtcnn(face_crop)
    if face is None:
        return None, None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = facenet(face).cpu().numpy()[0]  # (512,)

    return emb, [x1, y1, x2, y2]

def save_embeddings_and_centroids(new_embeddings, new_labels, new_label_map):
    """
    Save embeddings.pkl and centroids.pkl (centroids + label_map)
    """
    data = {
        "embeddings": np.array(new_embeddings),
        "labels": np.array(new_labels),
        "label_map": new_label_map
    }

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(data, f)

    # recompute centroids
    X = data["embeddings"]
    y = data["labels"]

    new_centroids = {}
    for lab in np.unique(y):
        lab = int(lab)
        new_centroids[lab] = X[y == lab].mean(axis=0)

    centroids_data = {
        "centroids": new_centroids,
        "label_map": new_label_map
    }

    with open(CENTROIDS_PATH, "wb") as f:
        pickle.dump(centroids_data, f)

    # update in-memory for instant use
    load_centroids()

# =========================
# INIT LOAD
# =========================
load_centroids()

# =========================
# API: Recognize (name + box)
# =========================
@app.route("/api/recognize", methods=["POST"])
def recognize_face():
    img = read_image_from_request()
    if img is None:
        return jsonify({"name": "Unknown"})

    emb, box = compute_embedding_from_image(img)
    if emb is None:
        return jsonify({"name": "Unknown"})

    # cosine similarity against centroids
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
        return jsonify({"name": "Unknown", "box": box})

    name = get_display_name(best_label)
    return jsonify({"name": name, "box": box})

# =========================
# API: Start training session
# =========================
@app.route("/api/start_session", methods=["POST"])
def start_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"name": "", "embeddings": []}
    return jsonify({"session_id": session_id})

# =========================
# API: Capture (store embedding in RAM)
# =========================
@app.route("/api/capture", methods=["POST"])
def capture_embedding():
    session_id = request.form.get("session_id", "").strip()
    person_name = request.form.get("name", "").strip()

    if not session_id or session_id not in sessions:
        return jsonify({"ok": False, "message": "Invalid session_id"}), 400

    if not person_name:
        return jsonify({"ok": False, "message": "Name required"}), 400

    img = read_image_from_request()
    if img is None:
        return jsonify({"ok": False, "message": "Image required"}), 400

    emb, box = compute_embedding_from_image(img)
    if emb is None:
        return jsonify({"ok": False, "message": "No face detected"}), 200

    sessions[session_id]["name"] = person_name
    sessions[session_id]["embeddings"].append(emb)

    return jsonify({
        "ok": True,
        "count": len(sessions[session_id]["embeddings"]),
        "box": box
    })

# =========================
# API: Train (write to embeddings.pkl & centroids.pkl)
# =========================
@app.route("/api/train", methods=["POST"])
def train_person():
    session_id = request.form.get("session_id", "").strip()
    person_name = request.form.get("name", "").strip()

    if not session_id or session_id not in sessions:
        return jsonify({"ok": False, "message": "Invalid session_id"}), 400

    if not person_name:
        return jsonify({"ok": False, "message": "Name required"}), 400

    embs = sessions[session_id]["embeddings"]
    if len(embs) < 3:
        return jsonify({"ok": False, "message": "Need at least 3 captures"}), 400

    # load existing embeddings.pkl
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, "rb") as f:
            data = pickle.load(f)
        X = list(data["embeddings"])
        y = list(data["labels"])
        lm = data.get("label_map", {})
    else:
        X, y, lm = [], [], {}

    # new label id
    if len(y) == 0:
        new_label = 0
    else:
        new_label = int(max(y)) + 1

    # append new embeddings
    for e in embs:
        X.append(e)
        y.append(new_label)

    lm[int(new_label)] = person_name

    # save and recompute centroids
    save_embeddings_and_centroids(X, y, lm)

    # clear RAM session
    sessions.pop(session_id, None)

    return jsonify({
        "ok": True,
        "message": "Training completed successfully",
        "label": int(new_label),
        "name": person_name
    })

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
