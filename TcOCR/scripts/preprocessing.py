import cv2
import os

def preprocess_image(image_path, output_path, target_size=(800, 600)):
    """
    Redimensionne et enregistre une image.
    """
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size)
    cv2.imwrite(output_path, resized_image)
    print(f"Image prétraitée enregistrée : {output_path}")

def preprocess_all_images(input_folder, output_folder):
    """
    Prétraite toutes les images d'un dossier.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            preprocess_image(input_path, output_path)


if __name__ == "__main__":
    input_folder = "../Data"  # Dossier contenant les ordonnances scannées
    output_folder = "../Data/Preprocessed"  # Dossier de sortie pour les images prétraitées
    preprocess_all_images(input_folder, output_folder)