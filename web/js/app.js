const video = document.getElementById("video");
const statusText = document.getElementById("status");
const detectedName = document.getElementById("detectedName");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const addBtn = document.getElementById("addBtn");
const captureBtn = document.getElementById("captureBtn");
const trainBtn = document.getElementById("trainBtn");

const nameInput = document.getElementById("personName");
const countInput = document.getElementById("imageCount");
const progressText = document.getElementById("progressText");
const progressFill = document.getElementById("progressFill");

let capturedImages = 0;
let requiredImages = 0;
let collecting = false;
let currentSessionId = null;

/* =========================
   START CAMERA
========================= */

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (error) {
    alert("Camera access denied");
  }
}

startCamera();

/* =========================
   CAPTURE FRAME
========================= */

function captureFrameBlob() {

  const tempCanvas = document.createElement("canvas");

  tempCanvas.width = video.videoWidth;
  tempCanvas.height = video.videoHeight;

  const tctx = tempCanvas.getContext("2d");

  tctx.drawImage(video, 0, 0);

  return new Promise(resolve => {
    tempCanvas.toBlob(blob => resolve(blob), "image/jpeg");
  });

}

/* =========================
   FACE RECOGNITION LOOP
========================= */

async function recognizeLoop() {

  if (video.readyState !== 4 || collecting) return;

  const imageBlob = await captureFrameBlob();

  const formData = new FormData();
  formData.append("image", imageBlob);

  try {

    const response = await fetch("http://localhost:5000/api/recognize", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    const displayWidth = video.clientWidth;
    const displayHeight = video.clientHeight;

    canvas.width = displayWidth;
    canvas.height = displayHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    statusText.innerText = data.name;
    detectedName.innerText = data.name;

    if (data.name === "Unknown") {
      statusText.className = "status unknown";
      addBtn.disabled = false;
    } else {
      statusText.className = "status known";
      addBtn.disabled = true;
      captureBtn.disabled = true;
      trainBtn.disabled = true;
    }

    if (data.box) {

      const [x1, y1, x2, y2] = data.box;

      const scaleX = displayWidth / video.videoWidth;
      const scaleY = displayHeight / video.videoHeight;

      const bx = x1 * scaleX;
      const by = y1 * scaleY;
      const bw = (x2 - x1) * scaleX;
      const bh = (y2 - y1) * scaleY;

      ctx.strokeStyle = data.name === "Unknown" ? "red" : "lime";
      ctx.lineWidth = 3;

      ctx.strokeRect(bx, by, bw, bh);

      ctx.fillStyle = ctx.strokeStyle;
      ctx.font = "18px Arial";
      ctx.fillText(data.name, bx, by - 10);
    }

  } catch {

    statusText.innerText = "Server not responding";

  }

}

setInterval(recognizeLoop, 1200);

/* =========================
   START TRAINING SESSION
========================= */

async function startTrainingSession() {

  const res = await fetch(
    "http://localhost:5000/api/start_session",
    { method: "POST" }
  );

  const data = await res.json();

  return data.session_id;

}

/* =========================
   ADD NEW PERSON
========================= */

addBtn.onclick = async () => {

  const name = nameInput.value.trim();

  if (!name) {
    alert("Please enter a name first");
    return;
  }

  requiredImages = parseInt(countInput.value, 10);

  if (isNaN(requiredImages) || requiredImages < 3) {
    alert("Enter a valid number of images (>=3)");
    return;
  }

  currentSessionId = await startTrainingSession();

  capturedImages = 0;
  collecting = true;

  progressText.innerText = `0 / ${requiredImages}`;
  progressFill.style.width = "0%";

  captureBtn.disabled = false;
  trainBtn.disabled = true;
  addBtn.disabled = true;

  statusText.innerText = "Image collection mode";

};

/* =========================
   CAPTURE IMAGE
========================= */

captureBtn.onclick = async () => {

  const name = nameInput.value.trim();

  if (!name || !currentSessionId) return;

  const imageBlob = await captureFrameBlob();

  const formData = new FormData();

  formData.append("session_id", currentSessionId);
  formData.append("name", name);
  formData.append("image", imageBlob);

  try {

    const res = await fetch(
      "http://localhost:5000/api/capture",
      {
        method: "POST",
        body: formData
      }
    );

    const data = await res.json();

    if (!data.ok) {

      alert(data.message || "Capture failed");
      return;

    }

    capturedImages = data.count;

    const percent = (capturedImages / requiredImages) * 100;

    progressText.innerText = `${capturedImages} / ${requiredImages}`;
    progressFill.style.width = percent + "%";

    if (capturedImages >= requiredImages) {

      captureBtn.disabled = true;
      trainBtn.disabled = false;
      collecting = false;

      alert("Images collected successfully. Now click Train Model");

    }

  } catch {

    alert("Server connection failed");

  }

};

/* =========================
   TRAIN MODEL
========================= */

trainBtn.onclick = async () => {

  const name = nameInput.value.trim();

  if (!name || !currentSessionId) return;

  statusText.innerText = "Training model...";
  statusText.className = "status known";

  const formData = new FormData();

  formData.append("session_id", currentSessionId);
  formData.append("name", name);

  try {

    const res = await fetch(
      "http://localhost:5000/api/train",
      {
        method: "POST",
        body: formData
      }
    );

    const data = await res.json();

    if (!data.ok) {

      alert(data.message || "Training failed");
      return;

    }

    alert("Training completed successfully");

    currentSessionId = null;
    capturedImages = 0;
    requiredImages = 0;

    progressText.innerText = "0 / 0";
    progressFill.style.width = "0%";

    addBtn.disabled = true;
    captureBtn.disabled = true;
    trainBtn.disabled = true;

    statusText.innerText = "Recognition mode";

  } catch {

    alert("Server error during training");

  }

};