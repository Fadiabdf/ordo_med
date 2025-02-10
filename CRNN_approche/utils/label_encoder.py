from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_vocabulary(labels):
    """Crée un vocabulaire à partir des étiquettes."""
    characters = set(''.join(labels))
    char_to_num = {char: idx for idx, char in enumerate(characters)}
    num_to_char = {idx: char for char, idx in char_to_num.items()}
    return char_to_num, num_to_char

def encode_labels(labels, char_to_num):
    """Encode les étiquettes en séquences d'indices."""
    encoded_labels = []
    for label in labels:
        encoded_label = [char_to_num[char] for char in label]
        encoded_labels.append(encoded_label)
    return pad_sequences(encoded_labels, padding='post')