import zipfile
import os
import random
import shutil

# 1. SETUP - Double check these paths!
ZIP_PATH = r"C:\Users\IDSL\face_recognition\vggface2.zip"
TARGET_DATASET = "research_sample"
IDENTITIES_TO_SELECT = 200 
IMAGES_PER_IDENTITY = 10

def smart_extract():
    if not os.path.exists(ZIP_PATH):
        print(f"Error: Could not find {ZIP_PATH}")
        return

    os.makedirs(TARGET_DATASET, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        print("Reading zip structure... (This takes a moment)")
        all_files = z.namelist()
        
        # Organize files by identity
        folder_structure = {}
        for f in all_files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                parts = f.split('/')
                if len(parts) >= 2:
                    identity = parts[-2] 
                    if identity not in folder_structure:
                        folder_structure[identity] = []
                    folder_structure[identity].append(f)

        identities = list(folder_structure.keys())
        print(f"Found {len(identities)} identities inside the zip.")

        # Pick our research subjects
        selected_ids = random.sample(identities, min(IDENTITIES_TO_SELECT, len(identities)))
        
        print(f"Extracting {IMAGES_PER_IDENTITY} images for {len(selected_ids)} people...")

        for identity in selected_ids:
            target_path = os.path.join(TARGET_DATASET, identity)
            os.makedirs(target_path, exist_ok=True)

            available_images = folder_structure[identity]
            selected_images = random.sample(available_images, min(IMAGES_PER_IDENTITY, len(available_images)))

            for img_path_in_zip in selected_images:
                filename = os.path.basename(img_path_in_zip)
                with z.open(img_path_in_zip) as source, open(os.path.join(target_path, filename), "wb") as target:
                    shutil.copyfileobj(source, target)

    print(f"\nSUCCESS: Research dataset ready in '{TARGET_DATASET}'!")

if __name__ == "__main__":
    smart_extract()