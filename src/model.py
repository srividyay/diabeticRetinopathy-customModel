import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
    
def simpleModel(input_shape=(512,512,3), num_classes=5):
    # Provided architecture

    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

class ModelWrapper:
    def __init__(self, model_path: str, class_names):
        self.model_path = model_path
        self.class_names = list(class_names)
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._model = tf.keras.models.load_model(self.model_path)

    def predict(self, batch):
        self._ensure_model()
        probs = self._model.predict(batch, verbose=0)
        indices = np.argmax(probs, axis=1)
        return indices, probs
