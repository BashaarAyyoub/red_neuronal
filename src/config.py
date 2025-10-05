from dataclasses import dataclass


@dataclass
class Config:
input_shape: int = 28 * 28
num_classes: int = 10
hidden_units: tuple = (128, 64)
activation: str = 'relu'
optimizer: str = 'adam'
loss: str = 'sparse_categorical_crossentropy'
metrics: tuple = ('accuracy',)
batch_size: int = 128
epochs: int = 5
output_dir: str = 'outputs'


DEFAULTS = Config()