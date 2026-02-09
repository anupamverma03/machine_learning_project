import cv2
import threading
import time
import pandas as pd
import numpy as np
from collections import deque
import os

# ====================== Threaded Camera Capture ======================
class VideoStream:
    def __init__(self, src=0, width=320, height=240):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()
            time.sleep(0.005)  # reduce CPU load

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


# ====================== Paths ======================
HAAR_PATH = "/home/anupam003/haarcascade_frontalface_default.xml"
PROTO_PATH = "/home/anupam003/models/deploy.prototxt"
MODEL_PATH = "/home/anupam003/models/res10_300x300_ssd_iter_140000.caffemodel"

# ====================== Load Models ======================
face_cascade = cv2.CascadeClassifier(HAAR_PATH)

net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ====================== Setup ======================
vs = VideoStream()
time.sleep(1.0)

roi = None
frame_count = 0
ssd_interval = 5
ssd_counter = 0

# Persistent detection + FPS smoothing
last_faces = []
fps_queue = deque(maxlen=20)
prev_time = time.time()
avg_fps = 0.0

# ---- Decay control (FIX) ----
miss_count = 0
MAX_MISSES = 6   # bounding box removed after 6 consecutive misses

# Logging
results = []
start_time = time.time()

# ====================== Main Loop ======================
while True:
    frame = vs.read()
    if frame is None:
        continue

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------- FPS calculation --------
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    fps_queue.append(fps)
    avg_fps = sum(fps_queue) / len(fps_queue)

    # -------- ROI logic --------
    if roi:
        x, y, w, h = roi
        gray_roi = gray[y:y+h, x:x+w]
    else:
        gray_roi = gray
        x, y = 0, 0

    faces = face_cascade.detectMultiScale(
        gray_roi,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )

    current_faces = []

    if len(faces) > 0:
        fx, fy, fw, fh = faces[0]

        roi = (
            max(0, x + fx - 20),
            max(0, y + fy - 20),
            min(frame.shape[1], fw + 40),
            min(frame.shape[0], fh + 40)
        )

        ssd_counter += 1
        if ssd_counter % ssd_interval == 0:
            face_crop = frame[
                roi[1]:roi[1] + roi[3],
                roi[0]:roi[0] + roi[2]
            ]

            if face_crop.size > 0:
                blob = cv2.dnn.blobFromImage(
                    face_crop, 1.0, (300, 300),
                    (104.0, 177.0, 123.0)
                )
                net.setInput(blob)
                detections = net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.6:
                        box = detections[0, 0, i, 3:7]
                        bx1 = int(box[0] * roi[2]) + roi[0]
                        by1 = int(box[1] * roi[3]) + roi[1]
                        bx2 = int(box[2] * roi[2]) + roi[0]
                        by2 = int(box[3] * roi[3]) + roi[1]
                        current_faces.append(
                            (bx1, by1, bx2 - bx1, by2 - by1)
                        )

        if current_faces:
            last_faces = current_faces.copy()
            miss_count = 0
        else:
            miss_count += 1

    else:
        roi = None
        miss_count += 1

    # -------- CLEAR STALE BOUNDING BOX (FIX) --------
    if miss_count >= MAX_MISSES:
        last_faces = []

    # -------- Draw bounding boxes --------
    for (x, y, w, h) in last_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # -------- Overlay (BLACK text) --------
    cv2.putText(
        frame,
        f"Avg FPS: {avg_fps:.2f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

    cv2.putText(
        frame,
        "Hybrid Haar + MobileNetSSD",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

    # -------- Logging --------
    results.append({
        "frame": frame_count,
        "avg_fps": round(avg_fps, 2),
        "faces": len(last_faces),
        "timestamp": round(time.time() - start_time, 3)
    })

    cv2.imshow("Stable Hybrid Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ====================== Cleanup ======================
vs.stop()
cv2.destroyAllWindows()

df = pd.DataFrame(results)
df.to_csv("hybrid_face_detection_log.csv", index=False)

print("✔ Results saved to hybrid_face_detection_log.csv")
print(f"✔ Avg FPS: {avg_fps:.2f}")
print(f"✔ Total frames: {frame_count}")
