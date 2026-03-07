import cv2
import os
import time

# =========================
# CONFIGURATION
# =========================
PERSON_NAME = "person_1"   # غيرها إلى person_2 للشخص الثاني
DATASET_PATH = "../dataset"
NUM_IMAGES = 400           # العدد المناسب أكاديميًا
CAPTURE_INTERVAL = 0.5     # ثانية بين كل صورة

# =========================
# CREATE SAVE PATH
# =========================
save_path = os.path.join(DATASET_PATH, PERSON_NAME)
os.makedirs(save_path, exist_ok=True)

# =========================
# OPEN CAMERA
# =========================
cap = cv2.VideoCapture(0)

count = len(os.listdir(save_path))
print(f"Starting from image count: {count}")
print("📸 الالتقاط تلقائي | اضغط 'q' للخروج")

last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ فشل في قراءة الكاميرا")
        break

    cv2.putText(
        frame,
        f"Images: {count}/{NUM_IMAGES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Automatic Face Data Collection", frame)

    current_time = time.time()

    # =========================
    # AUTO CAPTURE
    # =========================
    if current_time - last_capture_time >= CAPTURE_INTERVAL:
        img_name = f"img_{count:04d}.jpg"
        img_path = os.path.join(save_path, img_name)
        cv2.imwrite(img_path, frame)
        count += 1
        last_capture_time = current_time
        print(f"Saved: {img_name}")

        if count >= NUM_IMAGES:
            print("✅ تم جمع العدد الكافي من الصور")
            break

    # =========================
    # EXIT MANUALLY
    # =========================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🚪 خروج يدوي")
        break

cap.release()
cv2.destroyAllWindows()
