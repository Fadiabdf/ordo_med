
from ultralytics import YOLO
import cv2
import os

def segment_image(image_path, output_path, model_path="yolov8n.pt"):
    """
    Détecte les zones d'intérêt dans une image avec YOLO.
    """
    model = YOLO(model_path)
    results = model(image_path)

    # Enregistrer les résultats
    for result in results:
        result.save(output_path)
    print(f"Résultats de segmentation enregistrés : {output_path}")

def segment_all_images(input_folder, output_folder):
    """
    Segmente toutes les images d'un dossier.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            segment_image(input_path, output_path)


if __name__ == "__main__":
    input_folder = "../Data/Preprocessed"  # Dossier des images prétraitées
    output_folder = "../Data/Segmented"  # Dossier de sortie pour les images segmentées
    segment_all_images(input_folder, output_folder)