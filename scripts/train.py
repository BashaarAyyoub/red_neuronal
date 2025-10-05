from src.data import MNISTData
from src.model import MLPClassifier
from src.trainer import Trainer
from src.evaluate import Evaluator
from src.config import DEFAULTS
import os




def main():
cfg = DEFAULTS
os.makedirs(cfg.output_dir, exist_ok=True)


# 1) datos
data = MNISTData(flatten=True, normalize=True)
x_train, y_train, x_test, y_test = data.load()


# 2) modelo
model = MLPClassifier(input_dim=x_train.shape[1], hidden_units=cfg.hidden_units, num_classes=cfg.num_classes, activation=cfg.activation)
model.compile(optimizer=cfg.optimizer, loss=cfg.loss, metrics=cfg.metrics)


# 3) entrenador
trainer = Trainer(model, x_train, y_train, x_val=x_test, y_val=y_test, output_dir=cfg.output_dir)
history = trainer.train(epochs=cfg.epochs, batch_size=cfg.batch_size)


# 4) guardar modelo
model_path = trainer.save_model('mlp_mnist.h5')
print('Modelo guardado en:', model_path)


# 5) evaluación
evaluator = Evaluator(model)
cm = evaluator.confusion(x_test, y_test)
fig = evaluator.plot_confusion(cm)
fig.savefig(os.path.join(cfg.output_dir, 'confusion_matrix.png'))
print('Matriz de confusión guardada en:', os.path.join(cfg.output_dir, 'confusion_matrix.png'))




if __name__ == '__main__':
main()