import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    plt.close()

    plt.figure()
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()

def plot_confusion(y_true, y_pred, class_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()

def plot_confusion_by_dataset(y_true, y_pred, class_names, out_dir, dataset_name):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(out_dir, f'{dataset_name}_confusion_matrix.png'))
    plt.close()
