import tensorflow as tf
from tensorflow.keras import layers

def build_crnn(input_shape, num_classes, max_label_length):
    """Construit un modèle CRNN."""
    input_tensor = layers.Input(shape=input_shape, name='input')
    
    # Couches convolutives
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.MaxPooling2D((2, 2))(x)  # Sortie : (16, 64, 32)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # Sortie : (8, 32, 64)
    
    # Reshape pour les couches récurrentes
    # La sortie des couches convolutives a la forme (8, 32, 64)
    # Nous devons redimensionner en (max_label_length, 64)
    # Pour cela, nous devons ajuster la taille de la séquence
    x = layers.Reshape((-1, 64))(x)  # Redimensionne en (256, 64)
    
    # Couches récurrentes (LSTM bidirectionnelles)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    
    # Couche de sortie
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    # Vérifier la forme de la sortie
    print(f"Forme de la sortie du modèle : {x.shape}")  # Doit être (None, max_label_length, num_classes)
    model = tf.keras.Model(input_tensor, x, name='CRNN')
    return model