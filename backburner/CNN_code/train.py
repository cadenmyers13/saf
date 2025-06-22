import json

import model_setup
import numpy as np
import pandas as pd
import tensorflow as tf
import utils
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import CategoricalCrossentropy

# from keras_tuner import HyperParameters
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def compute_weights():
    weights = {}
    total_weights = size_sg.sum()[1]
    for i in range(45):
        weights[i] = total_weights / size_sg.loc[i, 1]
    return weights


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(
            tf.float32
        )  # I use ._decayed_lr method instead of .lr

    return lr


# hp = HyperParameters()
size_sg = pd.read_csv("size_sg.csv", header=None)
X_train = pd.read_csv("X_train.csv", header=None).to_numpy()
X_test = pd.read_csv("X_test.csv", header=None).to_numpy()
y_train = pd.read_csv("y_train.csv", header=None).to_numpy().flatten()
y_test = pd.read_csv("y_test.csv", header=None).to_numpy().flatten()

indices = utils.SG_ORDER_CNN
y_train_prime = np.array([np.where(indices == y)[0][0] for y in y_train])
y_test_prime = np.array([np.where(indices == y)[0][0] for y in y_test])
y_train_one_hot = tf.one_hot(y_train_prime, 45)
y_test_one_hot = tf.one_hot(y_test_prime, 45)

model = model_setup.PDF_CNN()
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(
        model_setup.lr_schedule, verbose=1
    )
]
# optimizer = Adam(hp.Float('learning_rate', min_value=1e-6, max_value=1e-3, sampling='LOG', default=5e-4))
# lr_metric = get_lr_metric(optimizer)
# model.compile(optimizer=optimizer,
model.compile(
    optimizer=Adam(),
    loss=CategoricalCrossentropy(),
    #                metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=6), lr_metric])
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=6)],
)
history = model.fit(
    X_train,
    y_train_one_hot,
    callbacks=callbacks,
    epochs=160,
    batch_size=64,
    class_weight=compute_weights(),
    validation_data=(X_test, y_test_one_hot),
)
model.save("my_model.h5")
history_dict = history.history
pd.DataFrame(history_dict).to_json("history.json")
