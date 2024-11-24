import dlib
import os
import json
from skimage import io
from tqdm import tqdm

# Input directory
image_dir = "/Users/cypherme/movie/full"

# Populate image paths
valid_extensions = {".jpg", ".jpeg", ".png"}
files = os.listdir(image_dir)
image_paths = [os.path.join(image_dir, f) for f in files if os.path.splitext(f)[1].lower() in valid_extensions]

# Initialize detector
detector = dlib.get_frontal_face_detector()

image_and_facenumber_pair_list = []
show_window = False

# Process images with progress bar
for path in tqdm(image_paths, desc="Processing images"):
    meta = {}
    image_id = os.path.splitext(os.path.basename(path))[0]
    meta['image_id'] = image_id

    try:
        img = io.imread(path)
        dets = detector(img, 1)
        meta['num_faces'] = len(dets)
        print(f"{len(dets)} faces were detected in image {path}.")
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        continue

    image_and_facenumber_pair_list.append(meta)

    if show_window:
        win = dlib.image_window()
        win.set_image(img)
        win.add_overlay(dets)
        input("Hit Enter to continue")

# Save results to JSON
output_file = os.path.join(image_dir, "image_and_facenumber_pair_list.json")
with open(output_file, "w") as f:
    json.dump(image_and_facenumber_pair_list, f)

print("Face detection for all images finished!")
