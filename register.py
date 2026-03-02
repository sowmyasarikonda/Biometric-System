import os
import cv2
import pickle
import time
from insightface.app import FaceAnalysis

app_ai = FaceAnalysis(name='buffalo_l') 
app_ai.prepare(ctx_id=0, det_size=(640, 640))

BASE_GALLERY_FOLDER = "anchor_gallery"
DATABASE_PATH = "database.pkl"

def save_new_user(user_id, img):
    
    safe_user_id = "".join(c for c in user_id if c.isalnum() or c in (' ', '_', '-')).strip()
    user_folder = os.path.join(BASE_GALLERY_FOLDER, safe_user_id)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    filename = f"face_{int(time.time())}.jpg"
    file_path = os.path.join(user_folder, filename)
    cv2.imwrite(file_path, img)

    faces = app_ai.get(img)
    if not faces:
        return {"success": False, "message": "No face detected in registration photo."}

    new_embedding = faces[0].normed_embedding

    db = {}
    if os.path.exists(DATABASE_PATH):
        try:
            with open(DATABASE_PATH, 'rb') as f:
                db = pickle.load(f)
        except Exception:
            db = {}

    db[safe_user_id] = new_embedding

    with open(DATABASE_PATH, 'wb') as f:
        pickle.dump(db, f)

    return {"success": True, "message": f"User {safe_user_id} added successfully!"}