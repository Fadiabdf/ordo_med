import os
import logging
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import json

# Désactiver les logs inutiles
logging.getLogger("transformers").setLevel(logging.ERROR)

def extract_text(image_path):
    """
    Extrait le texte d'une image avec TrOCR.
    """
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return extracted_text

def extract_text_from_all_images(input_folder, output_file):
    """
    Extrait le texte de toutes les images d'un dossier.
    """
    extracted_data = {}
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            text = extract_text(input_path)
            extracted_data[filename] = text
            print(f"Texte extrait de {filename}")

    # Enregistrer dans un fichier JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)
    print(f"Résultats enregistrés dans : {output_file}")

if __name__ == "__main__":
    input_folder = "../Data/Segmented"  # Dossier des images segmentées
    output_file = "../Data/extracted_text.json"  # Fichier de sortie pour le texte extrait
    extract_text_from_all_images(input_folder, output_file)