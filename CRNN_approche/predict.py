import numpy as np
from tensorflow.keras.models import load_model
from utils.data_loader import load_images
from utils.label_encoder import create_vocabulary

# Charger le modèle
model = load_model('models/crnn_model.h5')

# Charger les étiquettes pour créer le vocabulaire
train_csv = 'RIMES-2011-Lines/Train/train_labels.csv'
test_csv = 'RIMES-2011-Lines/Test/test_labels.csv'
train_filenames, train_labels = load_annotations(train_csv)
test_filenames, test_labels = load_annotations(test_csv)
char_to_num, num_to_char = create_vocabulary(train_labels + test_labels)

# Charger une image de test
test_image_dir = 'RIMES-2011-Lines/Test/images'
test_images = load_images(test_image_dir, [test_filenames[0]])  # Prédire sur la première image de test

# Faire une prédiction
predictions = model.predict(test_images)
decoded_text = ''.join([num_to_char[idx] for idx in np.argmax(predictions[0], axis=1)])
print(f"Texte prédit : {decoded_text}")