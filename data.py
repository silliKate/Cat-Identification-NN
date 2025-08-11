import os
import numpy as np
import h5py
from PIL import Image

# Folder with cat images
folder_path = "train_images/cats"
target_size = (64, 64)  # match old dataset size

# Step 1 — Load all images
images = []
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('RGB')  # ensure 3 channels
        img = img.resize(target_size)
        img_array = np.array(img)
        images.append(img_array)

# Convert to NumPy array
X = np.array(images)  # shape: (m, 64, 64, 3)
X = X.astype('float32') / 255.0  # normalize if you want

# Step 2 — Create labels
Y = np.ones((X.shape[0],))  # all cats → 1

# Step 3 — Save as .h5 file
with h5py.File("train_cats_only.h5", "w") as hf:
    hf.create_dataset("train_set_x", data=X)
    hf.create_dataset("train_set_y", data=Y)

print(f"Saved train_cats_only.h5 with {X.shape[0]} images.")
