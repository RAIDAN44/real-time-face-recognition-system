# Real-Time Face Recognition System

A real-time face recognition system built using **Deep Learning, Computer Vision, and Web Technologies**.

The system captures frames from a webcam, detects faces using **MTCNN**, generates embeddings using **FaceNet**, and identifies individuals by comparing embeddings with stored **centroids using cosine similarity**.

A web interface allows users to recognize faces in real time, add new people dynamically, and train the model directly from the browser.

---

# Features

* Real-time face recognition from webcam
* Face detection using **MTCNN**
* Face embeddings using **FaceNet**
* Identity prediction using **Cosine Similarity**
* Bounding box visualization
* Web-based interface
* Dynamic training directly from the UI
* Automatic centroid recomputation
* Flask REST API backend

---

# System Architecture

The recognition pipeline follows the standard deep learning workflow:

Camera Frame
↓
Face Detection (MTCNN)
↓
Face Embedding (FaceNet)
↓
Cosine Similarity Comparison
↓
Identity Recognition

---

# Technologies Used

### Programming Language

* Python

### Deep Learning

* PyTorch
* FaceNet (facenet-pytorch)

### Face Detection

* MTCNN

### Computer Vision

* OpenCV

### Machine Learning

* Scikit-learn

### Scientific Computing

* NumPy

### Web Backend

* Flask

### Frontend

* HTML
* CSS
* JavaScript

---

# Project Structure

```
face-recognition-system

dataset/                # training images (not included in repository)

models/                 # trained models
   embeddings.pkl
   centroids.pkl
   svm_classifier.pkl

server/
   app.py               # Flask backend API

src/                    # machine learning pipeline
   01_collect_data.py
   02_build_embeddings.py
   03_train_classifier.py
   04_realtime_test.py
   compute_centroids.py

web/                    # frontend interface
   index.html
   css/
   js/

run.py                  # start the full system
requirements.txt
README.md
```

---

# Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/face-recognition-system.git
cd face-recognition-system
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the System

Start the system using:

```
python run.py
```

Then open:

```
http://localhost:8000/web/index.html
```

---

# Face Recognition Pipeline

The system performs recognition in several steps:

1. Capture image frame from webcam
2. Detect faces using **MTCNN**
3. Extract embeddings using **FaceNet**
4. Compare embeddings with stored centroids
5. Identify the closest match using **cosine similarity**

---

# API Endpoints

### Recognize Face

```
POST /api/recognize
```

Response example:

```
{
  "name": "RAIDAN",
  "box": [120, 80, 240, 200]
}
```

---

### Start Training Session

```
POST /api/start_session
```

Returns a session ID for collecting new face embeddings.

---

### Capture Image

```
POST /api/capture
```

Stores embeddings temporarily in memory.

---

### Train Model

```
POST /api/train
```

Updates:

* embeddings.pkl
* centroids.pkl

---

# Future Improvements

* Multi-face tracking
* GPU acceleration
* Mobile camera support
* Cloud deployment
* Database integration

---

# Author

AI Student
Computer Vision Project
