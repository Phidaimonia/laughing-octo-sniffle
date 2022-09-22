
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
exp_name = "test_simple_dense"


callbacks = [TerminateOnNaN()]


# Architecture

with tf.device(DEVICE):

    K.clear_session()   
    


    class TransformerBlock(Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.05):
            super(TransformerBlock, self).__init__()

            self.num_heads = num_heads
            self.embed_dim = embed_dim
            self.ff_dim = ff_dim
            self.dropout_rate = rate

            self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, 
                                          kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0))
            self.ffn = Sequential(
                [Dense(ff_dim, activation="ReLU", kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0)),        # LeakyReLU
                 Dense(embed_dim, kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0)),]
            )
            self.layernorm1 = LayerNormalization(epsilon=1e-6)
            self.layernorm2 = LayerNormalization(epsilon=1e-6)
            self.dropout1 = Dropout(rate)
            self.dropout2 = Dropout(rate)

        @tf.function(jit_compile=False, experimental_follow_type_hints=True)
        def call(self, inputs, training=True):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output) # layernorm
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)       # layernorm2


        def get_config(self):
            cfg = super(TransformerBlock, self).get_config()
            cfg.update({'num_heads': self.num_heads,
                        'embed_dim': self.embed_dim,
                        'ff_dim': self.ff_dim,
                          'dropout_rate': self.dropout_rate})
            return cfg



    


    class PositionEmbedding(Layer):
        def __init__(self, maxlen, embed_dim):
            super(PositionEmbedding, self).__init__()
            self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)
            self.maxlen = maxlen

        @tf.function(jit_compile=False)
        def call(self, x):
            #maxlen = tf.shape(x)[-1]
            positions = tf.range(start=0, limit=self.maxlen, delta=1)
            positions = self.pos_emb(positions)
            return x + positions

        def get_config(self):
            cfg = super(PositionEmbedding, self).get_config()
            cfg.update({'pos_emb': self.pos_emb,
                        'maxlen': self.maxlen})
            return cfg

    
    
    
    
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
        
        
    




    class MEL_scale(Layer):
        def __init__(self, melYsize=128, frame_len=2048, samplingRate=16000):
            super(MEL_scale, self).__init__()
            self.melYsize = melYsize
            self.frame_len = frame_len
            self.samplingRate = samplingRate
            self.trainable = False
            
        def build(self, input_shape):
            self.melMatrix = tf.signal.linear_to_mel_weight_matrix(
                            num_mel_bins=self.melYsize,
                            num_spectrogram_bins=self.frame_len // 2 + 1,
                            sample_rate=self.samplingRate,
                            lower_edge_hertz=0,
                            upper_edge_hertz=self.samplingRate // 2)


        @tf.function(jit_compile=False)
        def call(self, x : tf.Tensor) -> tf.Tensor:

            return tf.matmul(tf.square(x), self.melMatrix)
        

        def get_config(self):
            cfg = super(MEL_scale, self).get_config()
            cfg.update({'melYsize': self.melYsize, 
                        'frame_len': self.frame_len,
                        'samplingRate': self.samplingRate})
            return cfg
        
        
        
    class Convert_dB_Norm(Layer):
        def __init__(self):
            super(Convert_dB_Norm, self).__init__()
            self.trainable = False

        @tf.function(jit_compile=False)
        def call(self, x : tf.Tensor) -> tf.Tensor:
            return (power_to_db(x) / 80.0) + 1.0
        

        
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
        
        

    
    class InvertedResidual(Layer):
        def __init__(self, filters, strides, expansion_factor=6, trainable=True,
                    name=None, **kwargs):
            super(InvertedResidual, self).__init__(trainable=trainable, name=name, **kwargs)
            self.filters = filters
            self.strides = strides
            self.expansion_factor = expansion_factor	# allowed to be decimal value
            self.activation = tf.nn.relu

        def build(self, input_shape):
            input_channels = int(input_shape[3])
            self.ptwise_conv1 = Conv2D(filters=int(input_channels*self.expansion_factor), kernel_size=1, use_bias=True, 
                                        kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0))
            self.dwise = DepthwiseConv2D(kernel_size=3, strides=self.strides, padding='same', use_bias=True, 
                                         kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0))
            self.ptwise_conv2 = Conv2D(filters=self.filters, kernel_size=1, use_bias=True, 
                                       kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0))

            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()
            self.bn3 = BatchNormalization()

        def call(self, input_x):
            # Expansion to high-dimensional space
            x = self.ptwise_conv1(input_x)
            x = self.bn1(x)
            x = self.activation(x)

            # Spatial filtering
            x = self.dwise(x)
            x = self.bn2(x)
            x = self.activation(x)

            # Projection back to low-dimensional space w/ linear activation
            x = self.ptwise_conv2(x)
            x = self.bn3(x)

            # Residual connection if i/o have same spatial and depth dims
            if input_x.shape[1:] == x.shape[1:]:
                x += input_x
            return x

        def get_config(self):
            cfg = super(InvertedResidual, self).get_config()
            cfg.update({'filters': self.filters,
                        'strides': self.strides,
                        'expansion_factor': self.expansion_factor})
            return cfg



    
    class ResidualBlock(Layer):
        def __init__(self, filters, trainable=True,
                    name=None, **kwargs):
            super(ResidualBlock, self).__init__(trainable=trainable, name=name, **kwargs)
            self.filters = filters
            self.activation = tf.nn.relu

        def build(self, input_shape):

            self.conv1 = Conv2D(filters=self.filters, kernel_size=3, padding="same", kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0))
            self.conv2 = Conv2D(filters=self.filters, kernel_size=3, padding="same", kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0))

            self.norm1 = BatchNormalization()
            self.norm2 = BatchNormalization()

        @tf.function(jit_compile=False)
        def call(self, input_x):

            x = self.conv1(input_x)
            x = self.norm1(x)
            x = self.activation(x)

            x = self.conv2(x)
            x = self.norm2(x)

            # Residual connection if we have same spatial and depth dims
            if input_x.shape[1:] == x.shape[1:]:
                x += input_x

            return self.activation(x)

        def get_config(self):
            cfg = super(ResidualBlock, self).get_config()
            cfg.update({'filters': self.filters})
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






x = Dense(64, activation='ReLU')(x)
x = Dense(64, activation='ReLU')(x)
x = Dense(16, activation='ReLU')(x)

x = Flatten()(x)

x = Dense(64, activation='ReLU')(x)
x = Dense(64, activation='ReLU')(x)



x = Dense(len(POSSIBLE_LABELS), activation = 'softmax', name='targets')(x)

outp = x






model = Model(inp, outp, name="model")
opt = LAMB(learning_rate=0.001)


loss = CategoricalCrossentropy(label_smoothing=0.1)
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])   # 'categorical_crossentropy'
model.build(input_shape=[timesteps, mel_bins])

model.summary()

gc.collect()



# %%

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



