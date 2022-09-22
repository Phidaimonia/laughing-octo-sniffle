
# %%
#%pip install pydub
#%pip install keras_tqdm
#%pip install tensorflow-addons
#%pip install tensorflow-io
#%pip install pandas


import tensorflow as tf

import tensorflow.keras as keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Conv2D, Flatten, MaxPooling2D, Conv1D, MaxPooling1D, Add, Concatenate, LocallyConnected1D
from keras.layers import Activation, BatchNormalization, GlobalMaxPooling1D, GlobalMaxPool2D
from keras.layers import Dense, Dropout, Reshape, LSTM, Layer, LayerNormalization, InputLayer, Permute, GRU, Cropping1D
from keras.layers import TimeDistributed, Conv2DTranspose, UpSampling2D, MultiHeadAttention, Embedding, Rescaling, Masking
from keras.layers import ZeroPadding1D, ZeroPadding2D, GaussianNoise, DepthwiseConv2D, Cropping2D, RepeatVector, RNN
from keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, AveragePooling2D
from keras.regularizers import l1, l2
from keras import activations, losses
from keras.constraints import max_norm

import tensorflow_addons as tfa
from tensorflow_addons.optimizers import LAMB 
import tensorflow_io as tfio

from keras import optimizers
from keras.losses import CategoricalCrossentropy
import numpy as np

import re
import os

import pandas as pd
from tensorflow.keras.utils import to_categorical


from random import *
import math

#import tqdm as tqdm
#from tqdm.notebook import tqdm, trange

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
import keras.backend as K
import matplotlib.pyplot as plt

import gc as gc


projDir = ''


#model_dtype = "float32"


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

tf.config.set_visible_devices([], 'GPU')       # uncomment to run on CPU

DEVICE = "/device:CPU:0"
print("Done")




# %%
# index dataset


POSSIBLE_LABELS = 'yes no up down left right on off stop go unknown'.split()      # 10 + unknown





id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}
len(POSSIBLE_LABELS)


#load dataset


trainX = np.load("specs_train.npy")
trainY = np.load("results_train.npy")
trainX = tf.convert_to_tensor(trainX, dtype=tf.float32) / 255.0

valX = np.load("specs_val.npy")
valY = np.load("results_val.npy")
valX = tf.convert_to_tensor(valX, dtype=tf.float32) / 255.0

print('There are {} train and {} validation (including background noise)'.format(trainX.shape[0], valX.shape[0]))



from keras.callbacks import TensorBoard
exp_name = "test_GRU"


callbacks = [TerminateOnNaN()]


# Architecture

with tf.device(DEVICE):

    K.clear_session()       
    
    
    
    class RandomNoise(Layer):
        def __init__(self, max_amp=0.1):
            super(RandomNoise, self).__init__()
            self.max_amp = max_amp
            self.trainable = False
            

        @tf.function(jit_compile=False)
        def call(self, x, training=None):
            inp_shape = tf.shape(x)
            batch_size = inp_shape[0]
            
            if not training:
                return x
            
            amp = tf.random.uniform([batch_size], minval=0.0, maxval=1.0, dtype=tf.float32) * self.max_amp  
            noise = tf.random.normal(shape=inp_shape, mean=0.0, stddev=1.0, dtype=tf.float32)
            
            amp = tf.reshape(amp, shape=[batch_size, 1, 1])
            amp = tf.tile(amp, multiples=(1, inp_shape[-2], inp_shape[-1]))        # expand the last dimension
            
            return x + noise * amp


        def get_config(self):
            cfg = super(RandomNoise, self).get_config()
            cfg.update({'max_amp': self.max_amp})
            return cfg



    
    class AugmentAmplitude(Layer):
        def __init__(self, mean_aug=0.1, percent_aug=0.1):
            super(AugmentAmplitude, self).__init__()
            self.mean_aug = mean_aug
            self.percent_aug = percent_aug
            self.trainable = False

        @tf.function(jit_compile=False)
        def call(self, x, training=None):

            inp_shape = tf.shape(x)
            batch_size = inp_shape[0]
            
            if not training:
                return x
            
            mean_shift = tf.random.uniform([batch_size], minval=-1.0, maxval=1.0, dtype=tf.float32) * self.mean_aug    
            # for mean_aug=0.1   -   (-0.1, 0.1)
            
            relative_change = tf.random.uniform([batch_size], minval=1.0-self.percent_aug, maxval=1.0+self.percent_aug, dtype=tf.float32)  
            # for 0.1 relative augmentation, get random on (0.9, 1.1)

    
            relative_change = tf.reshape(relative_change, shape=[batch_size, 1, 1])
            mean_shift = tf.reshape(mean_shift, shape=[batch_size, 1, 1])
            
            relative_change = tf.tile(relative_change, multiples=(1, inp_shape[-2], inp_shape[-1]))        # expand the last dimension
            mean_shift = tf.tile(mean_shift, multiples=(1, inp_shape[-2], inp_shape[-1])) 
            
            
            return x * relative_change + mean_shift 


        def get_config(self):
            cfg = super(AugmentAmplitude, self).get_config()
            cfg.update({'mean_aug': self.mean_aug, 
                        'percent_aug': self.percent_aug})
            return cfg
        
        
    




        
    class ClampLayer(Layer):
        def __init__(self, lower_bound=0.0, upper_bound=1.0):
            super(ClampLayer, self).__init__()
            self.lower = lower_bound
            self.upper = upper_bound
            self.trainable = False


        @tf.function(jit_compile=False)
        def call(self, x : tf.Tensor) -> tf.Tensor:
            return tf.clip_by_value(x, self.lower, self.upper)
        

        def get_config(self):
            cfg = super(ClampLayer, self).get_config()
            cfg.update({'lower': self.lower, 
                        'upper': self.upper})
            return cfg
        



# %%

################################ encoder



timesteps = 90
mel_bins = 60
embed_dim = 52



inp = Input(shape=[timesteps, mel_bins])      # Input = right from STFT

x = inp

x = AugmentAmplitude(mean_aug=0.05, percent_aug=0.05)(x) 
x = RandomNoise(0.05)(x)
x = ClampLayer(lower_bound=0.0, upper_bound=1.0)(x)





x = Dense(128, activation='ReLU', kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0))(x)
x = Dense(64, activation='ReLU', kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0))(x)

x = GRU(128, return_sequences=False, kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0))(x)


x = Dense(64, activation='ReLU', kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0))(x)
x = Dense(32, activation='ReLU', kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0))(x)



x = Dense(len(POSSIBLE_LABELS), activation = 'softmax', name='targets')(x)

outp = x






model = Model(inp, outp, name="model")
opt = LAMB(learning_rate=0.001)


loss = CategoricalCrossentropy(label_smoothing=0.1)
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])   # 'categorical_crossentropy'
model.build(input_shape=[timesteps, mel_bins])

model.summary()

gc.collect()







# SAM by sayakpaul@github
class SAMModel(tf.keras.Model):
    def __init__(self, orig_model, rho=0.05):
        """
        p, q = 2 for optimal results as suggested in the paper
        (Section 2)
        """
        super(SAMModel, self).__init__()
        self.orig_model = orig_model
        self.rho = rho

    @tf.function(jit_compile=False)
    def train_step(self, data):
        specs, labels, sample_weights = data
        
        e_ws = []
        with tf.GradientTape() as tape:
            predictions = self.orig_model(specs, training=True)
            loss = self.compiled_loss(labels, predictions, sample_weight=sample_weights)
			
        trainable_params = self.orig_model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        grad_norm = self._grad_norm(gradients)
        scale = self.rho / (grad_norm + 1e-12)

        for (grad, param) in zip(gradients, trainable_params):
            e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            predictions = self.orig_model(specs, training=True)
            loss = self.compiled_loss(labels, predictions, sample_weight=sample_weights)    
        
        sam_gradients = tape.gradient(loss, trainable_params)
        for (param, e_w) in zip(trainable_params, e_ws):
            param.assign_sub(e_w)
        
        self.optimizer.apply_gradients(
            zip(sam_gradients, trainable_params))
        
        self.compiled_metrics.update_state(labels, predictions, sample_weight=sample_weights)
        return {m.name: m.result() for m in self.metrics}



    def test_step(self, data):
        specs, labels = data
        predictions = self.orig_model(specs, training=False)
        loss = self.compiled_loss(labels, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, gradients):
        norm = tf.norm(
            tf.stack([
                tf.norm(grad) for grad in gradients if grad is not None
            ])
        )
        return norm
		
  
loss = CategoricalCrossentropy(label_smoothing=0.1)
SAM_model = SAMModel(model)
SAM_model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])





# %%
# train
gc.collect()

K.set_value(model.optimizer.learning_rate, 0.001)
batch_size = 1024




class_weights = np.sum(trainY) / (trainY.shape[1] * np.sum(trainY, axis=0))
class_weights = {i:w for i, w in enumerate(class_weights) }  # same as balanced from sklearn, n_samples / (n_classes * np.bincount(y)) 

print("Class weights:", class_weights)




try:
    hist = SAM_model.fit(x=trainX, y=trainY, validation_data=(valX, valY), callbacks=callbacks, class_weight=class_weights,
                     batch_size = batch_size, epochs = 100)  # 

    
    model.save_weights(exp_name)
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(hist.history["loss"], color="blue")
    plt.plot(hist.history["val_loss"], color="red")
    plt.title("Loss curve")
    plt.legend(["Training loss", "Validation loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig("loss_" + exp_name + ".eps") # eps 
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(hist.history["accuracy"], color="blue")
    plt.plot(hist.history["val_accuracy"], color="red")
    plt.title("Accuracy curve")
    plt.legend(["Training accuracy", "Validation accuracy"])
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.savefig("accuracy_" + exp_name + ".eps") 
    
    #plt.show()
except KeyboardInterrupt:
    print("Interrupted")



