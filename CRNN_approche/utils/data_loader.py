import os
import pandas as pd
import cv2
import numpy as np

def load_annotations(csv_file):
    """Charge les annotations depuis un fichier CSV."""
    df = pd.read_csv(csv_file)
    filenames = df['Filenames'].tolist()
    labels = df['Contents'].tolist()
    return filenames, labels

def load_images(image_dir, filenames):
    """Charge les images correspondantes."""
    images = []
    for filename in filenames:
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Image non trouvée : {image_path}")
            continue
        image = cv2.resize(image, (128, 32))
        image = image / 255.0
        image = np.expand_dims(image, axis=-1)
        images.append(image)
    return np.array(images)