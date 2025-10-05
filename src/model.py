from typing import Iterable, Optional, Tuple
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense




class MLPClassifier:
"""Encapsula un MLP con API de Keras.


Parameters
----------
input_dim : int
Dimensión de la entrada (784 para MNIST aplanado).
hidden_units : Iterable[int]
Lista o tupla con el número de neuronas por capa oculta.
num_classes : int
Número de clases de salida.
activation : str
Función de activación para capas ocultas.
"""


def __init__(self,
input_dim: int,
hidden_units: Iterable[int] = (128, 64),
num_classes: int = 10,
activation: str = 'relu'):
self.input_dim = input_dim
self.hidden_units = tuple(hidden_units)
self.num_classes = num_classes
self.activation = activation
self._model: Optional[Sequential] = None


def build(self) -> Sequential:
model = Sequential()
# primera capa con input_shape
model.add(Dense(self.hidden_units[0], activation=self.activation, input_shape=(self.input_dim,)))
for units in self.hidden_units[1:]:
model.add(Dense(units, activation=self.activation))
model.add(Dense(self.num_classes, activation='softmax'))


self._model = model
return model


def compile(self, optimizer: str = 'adam', loss: str = 'sparse_categorical_crossentropy', metrics: Tuple[str, ...] = ('accuracy',)):
if self._model is None:
self.build()
self._model.compile(optimizer=optimizer, loss=loss, metrics=list(metrics))


def fit(self, x, y, epochs: int = 5, batch_size: int = 32, validation_data=None, callbacks=None, verbose: int = 1):
if self._model is None:
raise RuntimeError('El modelo no está construido. Llama a build() o compile() primero.')
return self._model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=callbacks, verbose=verbose)


def evaluate(self, x, y, verbose: int = 1):
if self._model is None:
raise RuntimeError('Modelo no construido.')
return self._model.evaluate(x, y, verbose=verbose)


def predict(self, x):
if self._model is None:
raise RuntimeError('Modelo no construido.')
proba = self._model.predict(x)
import numpy as _np
return _np.argmax(proba, axis=-1)


def save(self, path: str):
os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
self._model.save(path)


@classmethod
def load(cls, path: str) -> 'MLPClassifier':
m = load_model(path)
# inferir dimensiones básicas de la red cargada
input_dim = int(m.input_shape[1])
num_classes = int(m.output_shape[1])
inst = cls(input_dim=input_dim, hidden_units=(), num_classes=num_classes)
inst._model = m
return inst