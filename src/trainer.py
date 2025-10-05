from typing import Optional
import os




class Trainer:
def __init__(self, model, x_train, y_train, x_val=None, y_val=None, output_dir: str = 'outputs'):
self.model = model
self.x_train = x_train
self.y_train = y_train
self.x_val = x_val
self.y_val = y_val
self.output_dir = output_dir
os.makedirs(self.output_dir, exist_ok=True)


def train(self, epochs: int = 5, batch_size: int = 128, callbacks: Optional[list] = None):
val = (self.x_val, self.y_val) if (self.x_val is not None and self.y_val is not None) else None
history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=val, callbacks=callbacks)
return history


def save_model(self, filename: str = 'mlp_model.h5'):
path = os.path.join(self.output_dir, filename)
self.model.save(path)
return path