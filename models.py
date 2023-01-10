import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

######25K models
def model1C25():
    input_im = tf.keras.Input(shape=(56, 72, 1))
    upscale = 8 # 48km to 3
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_im)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(upscale**2, (3,3), activation='relu', padding='same')(x)
    output = tf.nn.depth_to_space(x, upscale)
    
    return Model(input_im, output)

def model2C25():
    input_im = tf.keras.Input(shape=(56, 72, 1))
    upscale = 8
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_im)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(upscale**2, (3,3), activation='relu', padding='same')(x)
    output = tf.nn.depth_to_space(x, upscale)
    
    input_im2 = tf.keras.Input(shape=(448, 576, 1))
    cat = layers.Concatenate()([output, input_im2])
    
    z = layers.Conv2D(32, (3,3), activation='relu', padding='same')(cat)
    z = layers.Conv2D(16, (3,3), activation='relu', padding='same')(z)
    z = layers.Conv2D(1, (3,3), activation='relu', padding='same')(z)
    
    return Model([input_im, input_im2], z)


def model2C_timestamp25():
    input_im = tf.keras.Input(shape=(56, 72, 1))
    
    timestamp = tf.keras.Input(shape=(1,))
    time_embedded = layers.Embedding(96, 56*72, input_length=1)(timestamp)
    time_embedded = tf.reshape(time_embedded, [-1, 56, 72, 1])
    
    input_concat = layers.Concatenate()([input_im, time_embedded])
    
    upscale = 8
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_concat)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(upscale**2, (3,3), activation='relu', padding='same')(x)
    output = tf.nn.depth_to_space(x, upscale)
    
    input_im2 = tf.keras.Input(shape=(448, 576, 1))
    cat = layers.Concatenate()([output, input_im2])
    
    z = layers.Conv2D(32, (3,3), activation='relu', padding='same')(cat)
    z = layers.Conv2D(16, (3,3), activation='relu', padding='same')(z)
    z = layers.Conv2D(1, (3,3), activation='relu', padding='same')(z)
    
    return Model([input_im, timestamp, input_im2], z)
#### END 25K models

####48K models
def model1C48(): 
    input_im = tf.keras.Input(shape=(28, 36, 1))
    upscale = 16 # 48km to 3
    
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_im)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)

    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(upscale**2, (3,3), activation='relu', padding='same')(x)
    output = tf.nn.depth_to_space(x, upscale)
    
    return Model(input_im, output)

def model2C48():
    input_im = tf.keras.Input(shape=(28, 36, 1))
    upscale = 16
    
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_im)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(4**2, (3,3), activation='relu', padding='same')(x)
    
    
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(upscale**2, (3,3), activation='relu', padding='same')(x)
    output = tf.nn.depth_to_space(x, upscale)
    
    input_im2 = tf.keras.Input(shape=(448, 576, 1))
    cat = layers.Concatenate()([output, input_im2])
    
    z = layers.Conv2D(32, (3,3), activation='relu', padding='same')(cat)
    z = layers.Conv2D(16, (3,3), activation='relu', padding='same')(z)
    z = layers.Conv2D(1, (3,3), activation='relu', padding='same')(z)
    
    return Model([input_im, input_im2], z)

def model_2C_multioutput48():
    #input 1
    input_im  = tf.keras.Input(shape=(28,36,1))
    upscale1 = 4 #48 to 12
    ### change filters to constant 16 filters per layer
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_im)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(upscale1**2, (3,3), activation='relu', padding='same')(x)
    output1 = tf.nn.depth_to_space(x, upscale1) ###DONT COMPUTE MR12km LOSS HERE ###
    
    upscale2 = 4*4 #CHANGE THIS PART OF THE MODEL
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(upscale2**2, (3,3), activation='relu', padding='same')(x)
    output2 = tf.nn.depth_to_space(x, upscale2, name='output HR')

    #input 2
    input_im2 = tf.keras.Input(shape=(448,576,1)) # terrain image is input

    #concatenate models
    cat = layers.Concatenate()([output2, input_im2])

    #convolve again twice
    z = layers.Conv2D(32, (3,3), activation='relu', padding='same')(cat)
    z = layers.Conv2D(16, (3,3), activation='relu', padding='same')(z)
    z = layers.Conv2D(1, (3,3), activation='relu', padding='same')(z)

    return Model(inputs=[input_im, input_im2], outputs=[output1, z])

def model_timestamp48():
    input_im = tf.keras.Input(shape=(28, 36, 1))
    
    timestamp = tf.keras.Input(shape=(1,))
    time_embedded = layers.Embedding(96, 28*36, input_length=1)(timestamp)
    time_embedded = tf.reshape(time_embedded, [-1, 28, 36, 1])
    
    input_concat = layers.Concatenate()([input_im, time_embedded])
    
    upscale = 16
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_concat)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x) 
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(upscale**2, (3,3), activation='relu', padding='same')(x)
    output = tf.nn.depth_to_space(x, upscale)
    
    input_im2 = tf.keras.Input(shape=(448, 576, 1))
    cat = layers.Concatenate()([output, input_im2])
    
    z = layers.Conv2D(32, (3,3), activation='relu', padding='same')(cat)
    z = layers.Conv2D(16, (3,3), activation='relu', padding='same')(z)
    z = layers.Conv2D(1, (3,3), activation='relu', padding='same')(z)
    
    return Model([input_im, timestamp, input_im2], z)
####END 48K models


####100K models
def model_1C100(): #base 1 channel model, 32x upscale
    input_im  = tf.keras.Input(shape=(14,18,1))
    upscale1 = 4 #100 to 25
    ### change filters to constant 16 filters per layer
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_im)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    #x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    #might wanna change num filters between these two layers (no intuitive sense to go from 64 --> 16)
    x = layers.Conv2D(upscale1**2, (3,3), activation='relu', padding='same')(x)
    output1 = tf.nn.depth_to_space(x, upscale1) ###COMPUTE MR25km LOSS HERE, DON'T PASS TO NEXT CONV LAYER (56, 72)###
    
    upscale2 = 8*upscale1
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
#    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(upscale2**2, (3,3), activation='relu', padding='same')(x)
    output2 = tf.nn.depth_to_space(x, upscale2, name='output HR')
    
    return Model(input_im, output2)

def model_2C100():
    #model 1
    input_im  = tf.keras.Input(shape=(14,18,1))
    upscale1 = 4 #100 to 25
    ### change filters to constant 16 filters per layer
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_im)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(upscale1**2, (3,3), activation='relu', padding='same')(x)
    output1 = tf.nn.depth_to_space(x, upscale1) ###DONT COMPUTE MR25km LOSS HERE (56, 72)###
    
    upscale2 = 8*4
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(upscale2**2, (3,3), activation='relu', padding='same')(x)
    output2 = tf.nn.depth_to_space(x, upscale2, name='output HR')

    #model 2
    input_im2 = tf.keras.Input(shape=(448,576,1)) # terrain image is input
    
    #concatenate models
    cat = layers.Concatenate()([output2, input_im2])

    #convolve again twice
    z = layers.Conv2D(32, (3,3), activation='relu', padding='same')(cat)
    z = layers.Conv2D(16, (3,3), activation='relu', padding='same')(z)
    z = layers.Conv2D(1, (3,3), activation='relu', padding='same')(z)

    return Model(inputs=[input_im, input_im2], outputs=z)

def model_2C_multioutput100():
    #model 1
    input_im  = tf.keras.Input(shape=(14,18,1))
    upscale1 = 4 #100 to 25
    ### change filters to constant 16 filters per layer
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_im)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(upscale1**2, (3,3), activation='relu', padding='same')(x)
    output1 = tf.nn.depth_to_space(x, upscale1) ###COMPUTE MR25km LOSS HERE (56, 72)###
    
    upscale2 = 8*upscale1
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(upscale2**2, (3,3), activation='relu', padding='same')(x)
    output2 = tf.nn.depth_to_space(x, upscale2, name='output HR')

    #model 2
    input_im2 = tf.keras.Input(shape=(448,576,1)) # terrain image is input


    #concatenate models
    cat = layers.Concatenate()([output2, input_im2])

    #convolve again twice
    z = layers.Conv2D(32, (3,3), activation='relu', padding='same')(cat)
    z = layers.Conv2D(16, (3,3), activation='relu', padding='same')(z)
    z = layers.Conv2D(1, (3,3), activation='relu', padding='same')(z)

    return Model(inputs=[input_im, input_im2], outputs=[output1, z])

def mod_timestamp100():
    #model 1
    input_im  = tf.keras.Input(shape=(14,18,1))
    
    timestamp = tf.keras.Input(shape=(1,), dtype='int32')
    time_embedded = layers.Embedding(96, 14*18, input_length=1)(timestamp)
    time_embedded = tf.reshape(time_embedded, [-1, 14, 18, 1])
    
    input_concat = layers.Concatenate()([input_im, time_embedded])

    upscale1 = 4 #100 to 25
    ### change filters to constant 16 filters per layer
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_concat)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(upscale1**2, (3,3), activation='relu', padding='same')(x)
    #output1 = tf.nn.depth_to_space(x, upscale1) ###COMPUTE MR25km LOSS HERE (56, 72)###
    
    upscale2 = 8*upscale1
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(512, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(upscale2**2, (3,3), activation='relu', padding='same')(x)
    output2 = tf.nn.depth_to_space(x, upscale2, name='output HR')

    #model 2
    input_im2 = tf.keras.Input(shape=(448,576,1)) # terrain image is input

    #concatenate models
    cat = layers.Concatenate()([output2, input_im2])

    #convolve again twice
    z = layers.Conv2D(32, (3,3), activation='relu', padding='same')(cat)
    z = layers.Conv2D(16, (3,3), activation='relu', padding='same')(z)
    z = layers.Conv2D(1, (3,3), activation='relu', padding='same')(z)

    return Model(inputs=[input_im, timestamp, input_im2], outputs=z)
####END 100K models