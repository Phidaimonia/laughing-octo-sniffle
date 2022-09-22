
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




POSSIBLE_LABELS = 'yes no up down left right on off stop go unknown'.split()      # 10 + unknown





id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}
len(POSSIBLE_LABELS)



# %%
#load dataset


trainX = np.load("specs_train.npy")
trainY = np.load("results_train.npy")
trainX = tf.convert_to_tensor(trainX, dtype=tf.float32) / 255.0

valX = np.load("specs_val.npy")
valY = np.load("results_val.npy")
valX = tf.convert_to_tensor(valX, dtype=tf.float32) / 255.0

print('There are {} train and {} validation (including background noise)'.format(trainX.shape[0], valX.shape[0]))




# %%
from keras.callbacks import TensorBoard
exp_name = "MAE_large_tensorflow_200_probing"


callbacks = [TerminateOnNaN()]



# %%
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



    
    
    class RandomMask(Layer):
        def __init__(self, maxLen=64, masked_rate=0.75):
            super(RandomMask, self).__init__()

            self.maskedRate = masked_rate
            self.maxLen = maxLen
            self.trainable = False
 
        @tf.function(jit_compile=False)
        def call(self, inputs, training=None):
            
            batch_size = tf.shape(inputs)[-3]   # or 0 ?
            mask = tf.random.uniform(shape=(batch_size, self.maxLen,), minval=0.0, maxval=1.0, dtype=tf.float32)      # stateless_uniform
            mask = tf.cast(tf.math.greater(mask, self.maskedRate), dtype=tf.float32) 
            
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.tile(mask, multiples=(1, 1, tf.shape(inputs)[-1]))        # expand the last dimension
            
            #if training: 
            return tf.math.multiply(inputs, mask), mask
            
            #return inputs, tf.ones(tf.shape(inputs))        # ones = no mask


        def get_config(self):
            cfg = super(RandomMask, self).get_config()
            cfg.update({'maskedRate': self.maskedRate,
                        'maxLen': self.maxLen})
            return cfg
        
    
    class RestoreUnmaskedTokens(Layer):
        def __init__(self, maxLen=64):
            super(RestoreUnmaskedTokens, self).__init__()

            self.maxLen = maxLen
            self.trainable = False

        @tf.function(jit_compile=False)
        def call(self, inputs, training=None):
            
            reconstructed, original, mask = inputs
            
            rec = tf.math.multiply(reconstructed, 1.0 - mask)
            remain = tf.math.multiply(original, mask)
            
            #if training: 
            return rec + remain
            
            #return reconstructed


        def get_config(self):
            cfg = super(RestoreUnmaskedTokens, self).get_config()
            cfg.update({'maxLen': self.maxLen})
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
# Linear probing




timesteps = 90
mel_bins = 60
embed_dim = 128



inp = Input(shape=[timesteps, mel_bins])    

x = inp

before_mask = x
x, mask = RandomMask(maxLen=timesteps, masked_rate=0.0)(before_mask)

x = AugmentAmplitude(mean_aug=0.05, percent_aug=0.05)(x) 
x = RandomNoise(0.05)(x)
x = ClampLayer(lower_bound=0.0, upper_bound=1.0)(x)





x = Dense(256, activation = 'relu', kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0))(x)
x = Dense(128, activation = 'relu', kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0))(x)
x = PositionEmbedding(timesteps, 128)(x)
x = Dense(embed_dim, activation = 'relu', kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0))(x)


# Encoder
x = TransformerBlock(embed_dim, 4, embed_dim*2, rate=0.1)(x)
x = TransformerBlock(embed_dim, 4, embed_dim*2, rate=0.1)(x)
x = TransformerBlock(embed_dim, 4, embed_dim*2, rate=0.1)(x)
x = TransformerBlock(embed_dim, 4, embed_dim*2, rate=0.1)(x)

x = TransformerBlock(embed_dim, 4, embed_dim*2, rate=0.1)(x)
x = TransformerBlock(embed_dim, 4, embed_dim*2, rate=0.1)(x)
x = TransformerBlock(embed_dim, 4, embed_dim*2, rate=0.1)(x)
x = TransformerBlock(embed_dim, 4, embed_dim*2, rate=0.1)(x)




# Decoder
x = Dense(256, activation = 'relu', kernel_constraint=max_norm(10.0), bias_constraint=max_norm(10.0))(x)

x = GlobalAveragePooling1D()(x)


outp = x






encoder = Model(inp, outp, name="encoder")
opt = LAMB(learning_rate=0.001)

encoder.build(input_shape=[timesteps, mel_bins])


encoder.load_weights("weights/test_transformer_MAE_large").expect_partial()


#for i in range(17):
#    encoder.layers[i].trainable = False
    
encoder.trainable = False
encoder.summary()




# Classifier

inp = Input(shape=[256])    

x = inp

#x = Dense(256, activation = 'ReLU')(x) # 


x = Dense(len(POSSIBLE_LABELS), activation = 'softmax', name='targets')(x)

outp = x






classifier = Model(inp, outp, name="classifier")
opt = LAMB(learning_rate=0.001)

classifier.build(input_shape=[256])
classifier.summary()


# Merged model


inp = Input(shape=[timesteps, mel_bins])    

x = encoder(inp)
x = classifier(x)

outp = x






merged = Model(inp, outp, name="merged")
opt = LAMB(learning_rate=0.001)

loss = CategoricalCrossentropy(label_smoothing=0.1)
merged.compile(optimizer=opt, loss=loss, metrics=['accuracy'])  
merged.build(input_shape=[timesteps, mel_bins])
merged.summary()

gc.collect()



# %%
# train
gc.collect()

K.set_value(merged.optimizer.learning_rate, 0.001)
batch_size = 128


class_weights = np.sum(trainY) / (trainY.shape[1] * np.sum(trainY, axis=0))
class_weights = {i:w for i, w in enumerate(class_weights) }  # same as balanced from sklearn, n_samples / (n_classes * np.bincount(y)) 

print("Class weights:", class_weights)




try:

    hist = merged.fit(x=trainX, y=trainY, callbacks=callbacks, validation_data=(valX, valY), class_weight=class_weights,
                     batch_size = batch_size, epochs = 10)  # 
    
    #hist = SAM_model.fit(cur_dataset, validation_data=cur_val_dataset, callbacks=callbacks, class_weight=class_weights,
    #                 batch_size = batch_size, epochs = 100)  # 
    
    
    merged.save_weights("weights/" + exp_name + "_merged")
    encoder.save_weights("weights/" + exp_name + "_encoder")
    classifier.save_weights("weights/" + exp_name + "_classifier")
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(hist.history["loss"], color="blue")
    plt.plot(hist.history["val_loss"], color="red")
    plt.title("Loss curve")
    plt.legend(["Training loss", "Validation loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig("loss_" + exp_name + ".eps") # eps 
    
    #plt.show()
except KeyboardInterrupt:
    print("Interrupted")



