import cv2
import pickle
import time
import csv
import os
import numpy as np
import mediapipe as mp
from insightface.app import FaceAnalysis
from datetime import datetime

# Initialization
log_file = "latency_report.csv"

def initialize_log():
    try:
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "User", "Sim Score", "Blink (ms)", "AI Detect (ms)", "Search (ms)", "TOTAL (ms)"])
    except Exception as e:
        print(f"[Warning] Could not initialize log: {e}")

initialize_log()

# 1. Loading AI Models
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Database Handaling 
db_path = "database.pkl"
if os.path.exists(db_path):
    try:
        with open(db_path, "rb") as f:
            known_faces = pickle.load(f)
        print(f"Database loaded: {len(known_faces)} users found.")
    except Exception as e:
        print(f"Error loading database: {e}. Starting empty.")
        known_faces = {}
else:
    print("No database.pkl found. Running in detection-only mode.")
    known_faces = {}

all_names = list(known_faces.keys())
all_embeddings = np.array(list(known_faces.values())) if known_faces else np.empty((0, 512))

#Calculating Blink Ratio
def get_blink_ratio(landmarks, eye_indices):
    points = [landmarks[i] for i in eye_indices]
    v_dist = np.linalg.norm(np.array([points[1].x, points[1].y]) - np.array([points[5].x, points[5].y]))
    h_dist = np.linalg.norm(np.array([points[0].x, points[0].y]) - np.array([points[3].x, points[3].y]))
    return v_dist / h_dist

# Main Function
cap = cv2.VideoCapture(0)
authenticated = False
frame_counter = 0
last_faces = []
status_text = "System Initialized"

# Initialize metrics
blink_ms = 0.0
detect_ms = 0.0
search_ms = 0.0
total_ms = 0.0
max_sim = 0.0

print("System Running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Retrying...")
        time.sleep(0.1)
        continue
    
    frame_counter += 1
    t_start = time.perf_counter()

    try:
        # A. Blink Check
        t_blink_s = time.perf_counter() #blink calculation start
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ear = get_blink_ratio(face_landmarks.landmark, [33, 160, 158, 133, 153, 144])
                if ear < 0.2: 
                    authenticated = True
                    status_text = "Liveness Verified!"
        blink_ms = (time.perf_counter() - t_blink_s) * 1000

        # B. Face Detection - Every 3rd Frame
        t_detect_s = time.perf_counter()
        if frame_counter % 3 == 0:
            last_faces = app.get(frame)
        detect_ms = (time.perf_counter() - t_detect_s) * 1000

        # C. Identity Search
        if len(last_faces) == 0:
            t_search_start = time.perf_counter()
            name_display = "Scanning..."
            color = (100, 100, 100)
            all_matches_string = "None"
        
        for face in last_faces:
            if len(all_embeddings) > 0:
                # 1. Calculate scores for everyone
                similarities = np.dot(all_embeddings, face.normed_embedding)
                
                match_indices = np.where(similarities > 0.40)[0]
                
                if len(match_indices) > 0:
                    matched_names = [all_names[i] for i in match_indices]
                    
                    all_matches_string = "; ".join(matched_names)
                    
                    best_idx = np.argmax(similarities)
                    max_sim = similarities[best_idx]
                    
                    if authenticated:
                        name_display = f"MATCHED ({len(match_indices)} DB entries)"
                        ui_status = "Unlocked"
                        color = (0, 255, 0)
                    else:
                        # Person hasn't blinked yet
                        name_display = "BLINK TO UNLOCK"
                        ui_status = "Locked"
                        color = (255, 165, 0)
                else:
                    # Below 0.40 similarity score:
                    all_matches_string = "Unknown"
                    name_display = "UNKNOWN"
                    ui_status = "Access Denied"
                    color = (0, 0, 255)
                    max_sim = np.max(similarities) if len(similarities) > 0 else 0
            else:
                #if the database is empty
                name_display = "NO DB"
                all_matches_string = "No DB"
                color = (255, 255, 0)
                max_sim = 0.0

            # UI
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"{name_display} ({max_sim:.2f})", (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Time Stamp Logging
        if frame_counter % 15 == 0:
            total_ms = (time.perf_counter() - t_start) * 1000
            
            if last_faces and len(all_embeddings) > 0:
                search_ms = (time.perf_counter() - t_search_start)
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%H:%M:%S"), 
                        all_matches_string,
                        ui_status,       
                        f"{max_sim:.2f}",
                        f"{blink_ms:.1f}",
                        f"{detect_ms:.1f}",
                        f"{search_ms:.1f}", 
                        f"{total_ms:.1f}"
                    ])

    except Exception as e:
        print(f"Error in loop: {e}")

       
    cv2.imshow("Face Recognition System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()