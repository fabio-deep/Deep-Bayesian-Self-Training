'''
Author: Fabio De Sousa Ribeiro
E-mail: fdesousaribeiro@lincoln.ac.uk
Paper: Deep Bayesian Self-Training
arXiv URL: https://arxiv.org/pdf/1812.01681.pdf
journal URL: https://link.springer.com/article/10.1007/s00521-019-04332-4
'''

import os, sys, itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import *
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from tensorflow.python.eager import context
from tensorflow.keras.preprocessing.image import DirectoryIterator
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

class GeneratorFilenames(DirectoryIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.filenames_np = np.array(self.filenames)
        # split class name from image name class/image.jpg
        self.filenames_np = np.array([f.split('/')[1] for f in self.filenames])

    def _get_batches_of_transformed_samples(self, index_array):
        return (super()._get_batches_of_transformed_samples(index_array),
                self.filenames_np[index_array])

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        self.val_log_dir = os.path.join(log_dir, 'validation')
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

    def set_model(self, model):
        if context.executing_eagerly():
            self.val_writer = tf.contrib.summary.create_file_writer(self.val_log_dir)
        else:
            self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def _write_custom_summaries(self, step, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if 'val_' in k}
        if context.executing_eagerly():
            with self.val_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for name, value in val_logs.items():
                    tf.contrib.summary.scalar(name, value.item(), step=step)
        else:
            for name, value in val_logs.items():
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.val_writer.add_summary(summary, step)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not 'val_' in k}
        super(TrainValTensorBoard, self)._write_custom_summaries(step, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
       self.losses = []
       #self.lr = []

    def on_epoch_end(self, batch, logs={}):
       self.losses.append(logs.get('loss'))
       #self.lr.append(exp_decay(len(self.losses)))
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix:")
    else:
        print('Confusion matrix, without normalization:')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 fontsize=15,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual Label',fontsize=15)
    plt.xlabel('Predicted Label',fontsize=15)
    plt.tight_layout()

# import os
#
# train_datagen = ImageDataGenerator()
# train_generator = GeneratorFilenames(
#     directory='/home/fabio/Documents/Bayesian_Self-Training/data/GPU_1/MNIST_unlabelled/train_init',
#     image_data_generator=train_datagen,
#     target_size=(28, 28),
#     batch_size=32, shuffle=False, # remember to keep true
#     class_mode='categorical', color_mode='grayscale')
