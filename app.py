from flask import Flask, request, jsonify
from flask_cors import CORS 
import cv2
import numpy as np
import base64
import os
import pickle
import time
from datetime import datetime

from register import save_new_user
from verify import face_engine
from metrics_logger import MetricsLogger

logger = MetricsLogger()

app = Flask(__name__)
CORS(app)


recent_activity = []

@app.route('/verify', methods=['POST'])
def process_image():
    try:
        data = request.json
        base64_string = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(base64_string)

        np_arr = np.frombuffer(img_bytes, np.uint8)
        opencv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if opencv_img is None:
            return jsonify({"match": False, "message": "Decode failed"}), 400

        start_time = time.perf_counter()
        judgment = face_engine.verify(opencv_img)
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000

        logger.log(
            identity=judgment.get("name", "unknown"),
            similarity=judgment.get("score", 0),
            match=judgment.get("match", False),
            latency=latency) 

        if judgment['match']:
            log_entry = {
                "name": judgment['name'],
                "time": time.strftime('%H:%M:%S'),
                "score": judgment['score']
            }
            recent_activity.append(log_entry)
            
        return jsonify(judgment)

    except Exception as e:
        return jsonify({"match": False, "message": str(e)}), 500
    
@app.route('/api/register', methods=['POST'])
def register_new_face():
    data = request.json
    user_id = data['user_id']
    base64_string = data['image'].split(',')[1] if ',' in data['image'] else data['image']
    
    img_bytes = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    opencv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    result = save_new_user(user_id, opencv_img)
    return jsonify(result)

@app.route('/api/dashboard', methods=['GET'])
def get_stats():
    db_path = "database.pkl"
    last_update = "Never"
    user_count = 0
    users_list = []
    
    page = int(request.args.get('page', 1))
    search_query = request.args.get('search', '').strip().lower()
    per_page = 10
    
    if os.path.exists(db_path):
        mtime = os.path.getmtime(db_path)
        last_update = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        
        try:
            with open(db_path, 'rb') as f:
                db = pickle.load(f)
                user_count = len(db)  
                
                for user_id in db.keys():
                    str_id = str(user_id)
                    if search_query in str_id.lower():
                        users_list.append({
                            "user_id": str_id,
                            "date": "Stored in DB" 
                        })
        except Exception as e:
            print(f"[ERROR] Could not read database for dashboard: {e}")

    match_count = len(users_list)
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paginated_users = users_list[start_index:end_index]
    
    has_next = end_index < match_count

    return jsonify({
        "last_update": last_update,
        "user_count": user_count,        
        "match_count": match_count,       
        "recent_logs": recent_activity[-10:], # Keep the last 10 activities
        "users": paginated_users,        
        "current_page": page,
        "has_next": has_next
    })

@app.route('/api/delete_user', methods=['POST'])
def delete_user():
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({"success": False, "message": "User ID is required"}), 400

        db_path = "database.pkl"
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                db = pickle.load(f)
            
            if user_id in db:
                del db[user_id]
                with open(db_path, 'wb') as f:
                    pickle.dump(db, f)
                return jsonify({"success": True, "message": f"User {user_id} removed from AI memory."})
        
        return jsonify({"success": False, "message": "User not found in database."}), 404

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)