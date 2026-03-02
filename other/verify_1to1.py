import os
import cv2
import numpy as np
import time
from datetime import datetime
import csv
from insightface.app import FaceAnalysis 

# --- CONFIGURATION ---
REF_IMAGE_PATH = r"C:\Users\IDSL\Pictures\Camera Roll\Sowmya.jpg"
LOG_CSV_FILE = "face_metrics.csv"
SIMILARITY_THRESHOLD = 0.8
LEFT_EYE_INDICES = [35, 41, 42, 39, 37, 36]
RIGHT_EYE_INDICES = [89, 95, 96, 93, 91, 90]
EAR_THRESHOLD = 0.2
def calculate_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

# 1. Initialization
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(480, 480))


def main():
    detect_live_ms = 0.0
    blink_frames = 0
    total_blinks = 0
    ear_avg = 0.0
    liveness_ms = 0.0
    # 2. Prepare CSV for logging
    headers = ["timestamp", "similarity", "decision", "capture_ms", 
               "detect_ref_ms", "detect_live_ms","liveness_ms", "total_ms"]
    
    file_exists = os.path.exists(LOG_CSV_FILE)

    if not file_exists:
        with open(LOG_CSV_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    else:
        print(f"[INFO] Metrics file already exists: {LOG_CSV_FILE}")

    # 3. Process Reference Image (Only Once)
    print(f"[INFO] Processing reference image: {REF_IMAGE_PATH}")
    t_ref_start = time.perf_counter()
    
    ref_img = cv2.imread(REF_IMAGE_PATH)
    if ref_img is None:
        print(f"[ERROR] Reference image not found at {REF_IMAGE_PATH}")
        return

    ref_results = app.get(ref_img)
    if not ref_results:
        print("[ERROR] No face detected in reference image!")
        return


    ref_embedding = ref_results[0].normed_embedding
    detect_ref_ms = (time.perf_counter() - t_ref_start) * 1000 #reference image detection time

    # 4. Start Camera Loop
    cap = cv2.VideoCapture(0)
    print("[INFO] Camera started. Press 'q' to stop.")

    last_process_time = 0 
    process_interval = 0.5 #processing 2 frame per second
    
   
    decision = "Waiting..."
    similarity = 0.0
    color = (255, 255, 0)
    bbox = None 


    try:
        while True:
            t_total_start = time.perf_counter()

            # Metric: Capture Time
            t_cap_start = time.perf_counter()
            ret, frame = cap.read()
            capture_ms = (time.perf_counter() - t_cap_start) * 1000

            if not ret:
                break
            current_time = time.time()

            if (current_time - last_process_time) >= process_interval:
                last_process_time = current_time

                # Metric: Live Detection Time
                t_live_start = time.perf_counter()
                live_faces = app.get(frame)
                detect_live_ms = (time.perf_counter() - t_live_start) * 1000

                similarity = 0.0
                decision = "No Face"

                # 5. Face Comparison Logic
                if live_faces:
                    # Compare with reference image embeddings
                    live_embedding = live_faces[0].normed_embedding
                    
                    # Cosine Similarity Calculation
                    similarity = np.dot(ref_embedding, live_embedding)
                    t_liveness_start = time.perf_counter()
                    
                    if live_faces[0].landmark_2d_106 is not None:
                        landmarks = live_faces[0].landmark_2d_106
                        left_eye = landmarks[LEFT_EYE_INDICES]
                        right_eye = landmarks[RIGHT_EYE_INDICES]
                        
                        ear_avg = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
                        
                        if ear_avg < EAR_THRESHOLD:
                            blink_frames += 1
                        else:
                            if blink_frames >= 1:
                                total_blinks += 1
                            blink_frames = 0
                    liveness_ms = (time.perf_counter() - t_liveness_start) * 1000

                    if similarity >= SIMILARITY_THRESHOLD and total_blinks >=1:
                        decision = "Match"
                        color = (0, 255, 0)
                    elif similarity >= SIMILARITY_THRESHOLD and total_blinks ==0:
                        decision = "Please Blink"
                        color = (0,255,255)
                    else:
                        decision = "No Match"
                        color = (0, 0, 255)
                    bbox = live_faces[0].bbox.astype(int)

                else:
                    decision = "No Face"
                    similarity = 0.0
                    color = (255, 255, 0)
                    bbox = None

                # Metric: Total Time
                total_ms = (time.perf_counter() - t_total_start) * 1000
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                # 6. Log to CSV
                with open(LOG_CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, round(float(similarity), 4), decision, 
                                    round(capture_ms, 2), round(detect_ref_ms, 2),  
                                    round(detect_live_ms, 2),round(liveness_ms, 4), round(total_ms, 2)])

            # UI Display
            if bbox is not None:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, f"{decision} ({similarity:.2f})", (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            cv2.imshow("1:1 Verification", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 7. Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Process finished. Metrics successfully logged to {LOG_CSV_FILE}")

if __name__ == "__main__":
    main()