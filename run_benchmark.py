import os
import time
import pandas as pd
from verify import face_engine

# 1. SETUP
DATASET_DIR = "research_sample"
results = []

print("Starting Biometric Benchmark...")

# Get list of people
identities = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

# 2. RUN GENUINE TRIALS (Same person comparisons)
print(f"Testing Genuine Matches for {len(identities)} people...")
for person in identities:
    path = os.path.join(DATASET_DIR, person)
    images = [os.path.join(path, img) for img in os.listdir(path)]
    
    # Compare Image 1 with Image 2 of the same person
    if len(images) >= 2:
        img1 = images[0]
        img2 = images[1]
        
        start_time = time.time()
        # Call verification function
        judgment = face_engine.verify_two_images(img1, img2) 
        latency = time.time() - start_time
        
        results.append({
            "trial_type": "genuine",
            "score": judgment['score'],
            "latency": latency,
            "match": judgment['match']
        })

# 3. RUN IMPOSTOR TRIALS (Different people comparisons)
print("Testing Impostor Matches...")
for i in range(len(identities) - 1):
    personA = identities[i]
    personB = identities[i+1]
    
    imgA = os.path.join(DATASET_DIR, personA, os.listdir(os.path.join(DATASET_DIR, personA))[0])
    imgB = os.path.join(DATASET_DIR, personB, os.listdir(os.path.join(DATASET_DIR, personB))[0])
    
    start_time = time.time()
    judgment = face_engine.verify_two_images(imgA, imgB)
    latency = time.time() - start_time
    
    results.append({
        "trial_type": "impostor",
        "score": judgment['score'],
        "latency": latency,
        "match": judgment['match']
    })

# 4. SAVE DATA
df = pd.DataFrame(results)
df.to_csv("biometric_results.csv", index=False)
print("\nSUCCESS: results saved to biometric_results.csv")