import os
from utils.data_loader import load_annotations, load_images
from utils.label_encoder import create_vocabulary, encode_labels
from utils.losses import ctc_loss  # Importer la fonction de perte CTC
from models.crnn_model import build_crnn
from tensorflow.keras.optimizers import Adam

# Chemins des fichiers
train_csv = 'RIMES-2011-Lines/Train/train_labels.csv'
train_image_dir = 'RIMES-2011-Lines/Train/Images/'
test_csv = 'RIMES-2011-Lines/Test/test_labels.csv'
test_image_dir = 'RIMES-2011-Lines/Test/images'

# Charger les données
train_filenames, train_labels = load_annotations(train_csv)
test_filenames, test_labels = load_annotations(test_csv)
train_images = load_images(train_image_dir, train_filenames)
test_images = load_images(test_image_dir, test_filenames)

# Encoder les étiquettes
char_to_num, num_to_char = create_vocabulary(train_labels + test_labels)
train_encoded_labels = encode_labels(train_labels, char_to_num)
test_encoded_labels = encode_labels(test_labels, char_to_num)

# Après avoir chargé les données
print(f"Forme de train_images : {train_images.shape}")  # Doit être (batch_size, 32, 128, 1)
print(f"Forme de train_encoded_labels : {train_encoded_labels.shape}")  # Doit être (batch_size, max_label_length)
print(f"Forme de test_images : {test_images.shape}")  # Doit être (batch_size, 32, 128, 1)
print(f"Forme de test_encoded_labels : {test_encoded_labels.shape}")  # Doit être (batch_size, max_label_length)

# Construire et compiler le modèle
input_shape = (32, 128, 1)
num_classes = len(char_to_num)
max_label_length = 256  # Ajuster à la longueur de la séquence après les couches convolutives
crnn_model = build_crnn(input_shape, num_classes, max_label_length)

# Utiliser la fonction de perte CTC personnalisée
crnn_model.compile(optimizer=Adam(), loss=ctc_loss)

# Entraîner le modèle
crnn_model.fit(train_images, train_encoded_labels, epochs=10, batch_size=32, validation_data=(test_images, test_encoded_labels))

# Sauvegarder le modèle
crnn_model.save('models/crnn_model.h5')
