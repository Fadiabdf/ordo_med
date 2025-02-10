import tensorflow as tf

def ctc_loss(y_true, y_pred):
    """Fonction de perte CTC personnalisée."""
    # Vérifier les dimensions
    print(f"Forme de y_true : {y_true.shape}")  # Doit être (batch_size, max_label_length)
    print(f"Forme de y_pred : {y_pred.shape}")  # Doit être (batch_size, max_sequence_length, num_classes)
    
    # Longueur des séquences de prédiction
    input_length = tf.reduce_sum(tf.ones_like(y_pred[:, :, 0]), axis=1)
    
    # Longueur des étiquettes
    label_length = tf.reduce_sum(tf.ones_like(y_true), axis=1)
    
    # Ajuster la longueur de la séquence de prédiction
    if y_pred.shape[1] > y_true.shape[1]:
        y_pred = y_pred[:, :y_true.shape[1], :]  # Tronquer y_pred à la longueur de y_true
    
    # Calculer la perte CTC
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss