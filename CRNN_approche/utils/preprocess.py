import cv2
import numpy as np

def preprocess_image(image):
    """Prétraite une image pour le modèle CRNN."""
    image = cv2.resize(image, (128, 32))
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    return image