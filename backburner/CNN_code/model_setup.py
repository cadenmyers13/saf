from tensorflow import random
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Convolution1D, Dense, MaxPooling1D, Flatten, Dropout, Activation, average
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Model
import numpy as np

input_shape = (209, 1)
filters1 = 256
filters2 = 64
batch=32
kernel_size = 32
ens_models = 3
l2_lambda = 1e-5
dim = 209
num_classes = 45
hidden_dims = 128
drop = 0.5

from numpy.random import seed
seed(0)
random.set_seed(0)

def exp_decay(epoch):
   initial_lrate = 5e-4
   k = 0.025
   lrate = initial_lrate * np.exp(-k*epoch)
   return lrate

def lr_schedule(epoch):
  """Learning Rate Schedule
  Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
  Called automatically every epoch as part of callbacks during training.
  # Arguments
    epoch (int): The number of epochs
  # Returns
    lr (float32): learning rate
  """
  lr = 5e-4
  if epoch > 122:
    lr == 5e-6
  elif epoch > 81:
    lr == 5e-5
  print('Learning rate: ', lr)
  return lr


def PDF_CNN(input_shape=input_shape,
            num_filters1=filters1,
            num_filters2=filters2,
            kernel_size=kernel_size,
            ens_models=ens_models,
            l2_lambda=l2_lambda,
            input_dims=dim,
            hidden_dims=hidden_dims,
            drop=drop,
            num_classes=num_classes):
  outs = []
  inputs = Input(shape=input_shape)
  inp_bn = BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None)(inputs)
  for i in range(ens_models):
    conv1 = Convolution1D(num_filters1,
                         kernel_size, padding='same',
                         activation='relu',
	                       kernel_initializer='he_uniform',
	                       kernel_regularizer=regularizers.l2(l2_lambda))(inp_bn)
    bn1 = BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None)(conv1)
    conv2 = Convolution1D(num_filters2,
                         kernel_size, padding='same',
                         activation='relu',
  	                     kernel_initializer='he_uniform',
  	                     kernel_regularizer=regularizers.l2(l2_lambda))(bn1)
    bn2 = BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None)(conv2)
    pool = MaxPooling1D()(bn2)
    drop1 = Dropout(drop)(pool)
    flat = Flatten()(drop1)
    dense = Dense(hidden_dims, activation='relu',
                  kernel_regularizer=regularizers.l2(l2_lambda),
                  kernel_initializer='he_uniform')(flat)
    bn3 = BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None)(dense)
    drop2 = Dropout(drop)(bn3)
    out = Dense(num_classes, activation='softmax',
                kernel_regularizer=regularizers.l2(l2_lambda),
                kernel_initializer='glorot_uniform')(drop2)
    outs.append(out)

  if ens_models > 1:
    outputs = average(outs)
  else:
    outputs = out
  model = Model(inputs=inputs, outputs=outputs)

  return model