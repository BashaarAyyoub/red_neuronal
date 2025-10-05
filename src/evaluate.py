from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import itertools




class Evaluator:
def __init__(self, model):
self.model = model


def confusion(self, x_test, y_test):
y_pred = self.model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
return cm


def plot_confusion(self, cm, labels=None, figsize=(8, 6)):
if labels is None:
labels = list(range(cm.shape[0]))
fig, ax = plt.subplots(figsize=figsize)
cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_title('Matriz de confusión')
tick_marks = np.arange(len(labels))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.xlabel('Predicción')
plt.ylabel('Etiqueta real')
fig.colorbar(cax)


thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
ax.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")


plt.tight_layout()
return fig


def classification_report(self, x_test, y_test):
y_pred = self.model.predict(x_test)
return classification_report(y_test, y_pred)