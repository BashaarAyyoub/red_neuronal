"""Carga y preprocesamiento de datos (MNIST).


Proporciona una clase que devuelve los arrays listos para entrenar.
"""
from typing import Tuple
import numpy as np
from tensorflow.keras.datasets import mnist




class MNISTData:
"""Clase para cargar y preprocesar MNIST.


Parámetros
----------
flatten : bool
Si True, convierte las imágenes 28x28 en vectores de 784 elementos.
normalize : bool
Si True, escala los píxeles a [0, 1].
"""


def __init__(self, flatten: bool = True, normalize: bool = True):
self.flatten = flatten
self.normalize = normalize


def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
"""Carga MNIST y aplica preprocesamiento básico.


Returns
-------
x_train, y_train, x_test, y_test
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()


if self.normalize:
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


if self.flatten:
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
else:
# añadir canal si se mantiene la forma 2D
x_train = x_train[..., None]
x_test = x_test[..., None]


return x_train, y_train, x_test, y_test