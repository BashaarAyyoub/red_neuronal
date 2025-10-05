import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Cargar el modelo CNN entrenado en Colab
model = tf.keras.models.load_model("model/cnn_model.h5")

def predict_digit(img_path):
    """Procesa la imagen y devuelve la predicción del modelo CNN"""
    img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 28, 28, 1)
    
    prediction = model.predict(img_array)
    result = np.argmax(prediction)
    return int(result)
