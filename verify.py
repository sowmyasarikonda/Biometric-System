import os
import pickle
import numpy as np
from insightface.app import FaceAnalysis


class FaceLogic:
    def __init__(self, db_path="database.pkl"):
        self.db_path = db_path
        self.database = {}
        self.last_loaded_time = 0
        
        # Initialize InsightFace
        self.app_ai = FaceAnalysis(name='buffalo_l')
        self.app_ai.prepare(ctx_id=0, det_size=(640, 640))
        
        self.load_db()

    def load_db(self):
        #Reloads the database only if the file on disk is newer
        if os.path.exists(self.db_path):
            current_mtime = os.path.getmtime(self.db_path)
            if current_mtime > self.last_loaded_time:
                try:
                    with open(self.db_path, 'rb') as f:
                        self.database = pickle.load(f)
                    self.last_loaded_time = current_mtime
                    print(f"[INFO] Database hot-reloaded. {len(self.database)} users active.")
                except Exception as e:
                    print(f"[ERROR] Failed to load database: {e}")

    def verify(self, img):
        # Trigger the auto-reload check
        self.load_db()
        
        if not self.database:
            return {"match": False, "name": "Unknown", "score": 0.0, "message": "Database empty."}

        # Get embeddings for the current face
        faces = self.app_ai.get(img)
        if not faces:
            return {"match": False, "name": "Unknown", "score": 0.0, "message": "No face detected."}

        current_embedding = faces[0].normed_embedding
        best_match = "Unknown"
        high_score = 0.0

        # Math comparison
        for name, saved_embedding in self.database.items():
            score = np.dot(current_embedding, saved_embedding)
            if score > high_score:
                high_score = float(score)
                best_match = name

        threshold = 0.8
        if high_score > threshold:
            return {"match": True, "name": best_match, "score": round(high_score, 2), "message": "Access Granted"}
        
        return {"match": False, "name": "Unknown", "score": round(high_score, 2), "message": "Unknown Face"}

face_engine = FaceLogic()