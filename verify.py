import os
import pickle
import numpy as np
from insightface.app import FaceAnalysis
import cv2

class FaceLogic:
    def __init__(self, db_path="database.pkl"):
        self.db_path = db_path
        self.database = {}
        self.last_loaded_time = 0
        
        # Initialize InsightFace
        self.app_ai = FaceAnalysis(name='buffalo_l')
        self.app_ai.prepare(ctx_id=-1, det_size=(640, 640))
        
        self.load_db()

    def load_db(self):
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
        self.load_db()
        
        if not self.database:
            return {"match": False, "name": "Unknown", "score": 0.0, "message": "Database empty."}

        faces = self.app_ai.get(img)
        if not faces:
            return {"match": False, "name": "Unknown", "score": 0.0, "message": "No face detected."}

        current_embedding = faces[0].normed_embedding
        best_match = "Unknown"
        high_score = 0.0

        for name, saved_embedding in self.database.items():
            score = np.dot(current_embedding, saved_embedding)
            if score > high_score:
                high_score = float(score)
                best_match = name

        threshold = 0.8
        if high_score > threshold:
            return {"match": True, "name": best_match, "score": round(high_score, 2), "message": "Access Granted"}
        
        return {"match": False, "name": "Unknown", "score": round(high_score, 2), "message": "Unknown Face"}

    def verify_two_images(self, path1, path2):
        """
        Special helper for research benchmarking.
        Uses self.app_ai to extract embeddings from two image files.
        """
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        if img1 is None or img2 is None:
            return {'match': False, 'score': 0.0}

        # Use self.app_ai (the name defined in __init__)
        faces1 = self.app_ai.get(img1)
        faces2 = self.app_ai.get(img2)

        # Safety check: ensure both images have a detectable face
        if not faces1 or not faces2:
            return {'match': False, 'score': 0.0}

        # Extract normalized embeddings
        feat1 = faces1[0].normed_embedding
        feat2 = faces2[0].normed_embedding
        
        # Calculate Cosine Similarity via Dot Product
        score = float(np.dot(feat1, feat2))
        
        # In research, we keep the score even if match is False 
        # so we can calculate EER/FAR/FRR later
        return {
            'match': bool(score > 0.4), 
            'score': score
        }

face_engine = FaceLogic()