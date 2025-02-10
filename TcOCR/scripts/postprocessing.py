# scripts/postprocessing.py
import re

def clean_text(text):
    """
    Nettoie le texte extrait (supprime les espaces inutiles, les caractères spéciaux, etc.).
    """
    text = re.sub(r"\s+", " ", text)  # Supprime les espaces multiples
    text = text.strip()  # Supprime les espaces en début et fin
    return text

def process_extracted_text(input_file, output_file):
    """
    Nettoie et structure les données extraites.
    """
    with open(input_file, "r") as f:
        lines = f.readlines()

    cleaned_data = []
    for line in lines:
        cleaned_line = clean_text(line)
        cleaned_data.append(cleaned_line)

    with open(output_file, "w") as f:
        f.writelines(cleaned_data)
    print(f"Données nettoyées enregistrées dans : {output_file}")

if __name__ == "__main__":
    input_file = "../Data/extracted_text.txt"  # Fichier de texte extrait
    output_file = "../Data/cleaned_text.txt"  # Fichier de sortie pour le texte nettoyé
    process_extracted_text(input_file, output_file)