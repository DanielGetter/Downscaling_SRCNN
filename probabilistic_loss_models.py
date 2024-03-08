import tensorflow as tf
# import tensorflow_probability as tfp
#import keras.backend as K
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import math as m
import numpy as np

#custom callbacks
checkpoint_callback1 = tf.keras.callbacks.ModelCheckpoint(filepath='MRHR25kmodel_checkpoints/3channel_gumbel/model_epoch:{epoch:02d}.h5', save_best_only=False, save_weights_only=False, monitor='val_loss', save_freq=890, verbose=1)

cc_lin_mu = tf.keras.callbacks.ModelCheckpoint(filepath='MRHR25kmodel_checkpoints/3channel_gumbel_linear_full_loss/model_epoch:{epoch:02d}.h5', save_best_only=False, save_weights_only=False, monitor='val_loss', save_freq=890, verbose=1)

cc_alllin = tf.keras.callbacks.ModelCheckpoint(filepath='MRHR25kmodel_checkpoints/3channel_gumbel_linear/model_epoch:{epoch:02d}.h5', save_best_only=False, save_weights_only=False, monitor='val_loss', save_freq=890, verbose=1)

cc_log_gam = tf.keras.callbacks.ModelCheckpoint(filepath='MRHR25kmodel_checkpoints/3channel_gumbel_log_gamma/model_epoch:{epoch:02d}.h5', save_best_only=False, save_weights_only=False, monitor='val_loss', save_freq=890, verbose=1)

checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(filepath='MRHR48kmodel_checkpoints/3channel_gumbel/model_epoch:{epoch:02d}.h5', save_best_only=False, save_weights_only=False, monitor='val_loss', save_freq=890, verbose=1)

checkpoint_callback3 = tf.keras.callbacks.ModelCheckpoint(filepath='LRHRmodel_checkpoints/3channel_gumbel/model_epoch:{epoch:02d}.h5', save_best_only=False, save_weights_only=False, monitor='val_loss', save_freq=890, verbose=1)

#test for using relu activation for sigma layer 1C
# checkpoint_callback4 = tf.keras.callbacks.ModelCheckpoint(filepath='MRHR25kmodel_checkpoints/3channel_gumbel_test/model_epoch:{epoch:02d}.h5', save_best_only=False, save_weights_only=False, monitor='val_loss', save_freq=890, verbose=1)

checkpoint_callback5 = tf.keras.callbacks.ModelCheckpoint(filepath='MRHR25kmodel_checkpoints/3channel_gauss/model_epoch:{epoch:02d}.h5', save_best_only=False, save_weights_only=False, monitor='val_loss', save_freq=890, verbose=1)

checkpoint_callback6 = tf.keras.callbacks.ModelCheckpoint(filepath='MRHR48kmodel_checkpoints/3channel_gauss/model_epoch:{epoch:02d}.h5', save_best_only=False, save_weights_only=False, monitor='val_loss', save_freq=890, verbose=1)

checkpoint_callback7 = tf.keras.callbacks.ModelCheckpoint(filepath='LRHRmodel_checkpoints/3channel_gauss/model_epoch:{epoch:02d}.h5', save_best_only=False, save_weights_only=False, monitor='val_loss', save_freq=890, verbose=1)

random_seed = 123
tf.random.set_seed(random_seed)
np.random.seed(random_seed)


def sin_95(x):
    return np.round(np.sin(((2*np.pi)/95)*x), 5)

def elu_plus_one_activation(x):
    return 1.0 + tf.keras.activations.elu(x)

d = np.load('split_data_US.npz')
xtrainHR, xtrainMR, xtrainMR2, xtrainMR12, xtrainLR, tertrain, timetrain, xtestHR, xtestMR, xtestMR2, xtestMR12, xtestLR, tertest, timetest = d['name1'], d['name2'], d['name3'], d['name4'], d['name5'], d['name6'], d['name7'], d['name8'], d['name9'], d['name10'], d['name11'], d['name12'], d['name13'], d['name14']
#transform data#
xtrainHR, xtrainMR, xtrainLR, tertrain, xtestHR, xtestMR, xtestLR, tertest = np.sqrt(xtrainHR), np.sqrt(xtrainMR), np.sqrt(xtrainLR), np.sqrt(tertrain+1), np.sqrt(xtestHR), np.sqrt(xtestMR), np.sqrt(xtestLR), np.sqrt(tertest+1) 
xtrainMR2, xtestMR2 = np.sqrt(xtrainMR2), np.sqrt(xtestMR2)
xtrainMR12, xtestMR12 = np.sqrt(xtrainMR12), np.sqrt(xtestMR12)
timetrain, timetest = sin_95(timetrain), sin_95(timetest)

#-------------------------------------------------------------------------------------------------
# MODELS -----------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

def mod25(loss, learning_rate=.0001):
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
    
    #pdf parameter final output nodes
    input_truth = tf.keras.Input(shape=(448,576, 1), name='input_truth')
    
    mu_pred = layers.Conv2D(1, (3,3), activation='relu', padding='same')(z) 
    sigma_pred = layers.Conv2D(1, (3,3), activation=elu_plus_one_activation, padding='same')(z)
    
    model = Model(inputs=[input_im, timestamp, input_im2, input_truth], outputs=[mu_pred, sigma_pred])
    model.add_loss(loss(input_truth, mu_pred, sigma_pred))
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=None)
    
    return model

def mod25_alllin(loss, learning_rate=.0001):
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
    
    #pdf parameter final output nodes
    input_truth = tf.keras.Input(shape=(448,576, 1), name='input_truth')
    
    mu_pred = layers.Conv2D(1, (3,3), activation='linear', padding='same')(z) 
    sigma_pred = layers.Conv2D(1, (3,3), activation=elu_plus_one_activation, padding='same')(z)
    
    model = Model(inputs=[input_im, timestamp, input_im2, input_truth], outputs=[mu_pred, sigma_pred])
    model.add_loss(loss(input_truth, mu_pred, sigma_pred))
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=None)
    
    return model

def mod25_log_gamma(loss, learning_rate=.0001):
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

    #pdf parameter final output nodes
    input_truth = tf.keras.Input(shape=(448,576, 1), name='input_truth')

    mu_pred = layers.Conv2D(1, (3,3), activation='linear', padding='same')(z)
    sigma_pred = layers.Conv2D(1, (3,3), activation=elu_plus_one_activation, padding='same')(z)

    model = Model(inputs=[input_im, timestamp, input_im2, input_truth], outputs=[mu_pred, sigma_pred])
    model.add_loss(loss(input_truth, mu_pred, sigma_pred))
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=None)

    return model



def mod48(loss, learning_rate=.0001):
    input_im = tf.keras.Input(shape=(28, 36, 1))
    
    timestamp = tf.keras.Input(shape=(1,))
    time_embedded = layers.Embedding(96, 28*36, input_length=1)(timestamp)
    time_embedded = tf.reshape(time_embedded, [-1, 28, 36, 1])
    
    input_concat = layers.Concatenate()([input_im, time_embedded])
    
    upscale = 16
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_concat)
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
    
    #pdf parameter layers
    input_truth = tf.keras.Input(shape=(448,576,1), name='wind_true')
    mu_pred = layers.Conv2D(1, (3,3), activation='relu', padding='same')(z) #change to relu if results subpar
    sigma_pred = layers.Conv2D(1, (3,3), activation=elu_plus_one_activation, padding='same')(z)
    
    model = Model(inputs=[input_im, timestamp, input_im2, input_truth], outputs=[mu_pred, sigma_pred])
    model.add_loss(loss(input_truth, mu_pred, sigma_pred))
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=None) 

    return model


def mod100(loss, learning_rate=.0001):
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
    
    #pdf parameter layers
    input_truth = tf.keras.Input(shape=(448,576,1), name='wind_true')
    mu_pred = layers.Conv2D(1, (3,3), activation='relu', padding='same')(z) #change to relu if results subpar
    sigma_pred = layers.Conv2D(1, (3,3), activation=elu_plus_one_activation, padding='same')(z)

    model = Model(inputs=[input_im, timestamp, input_im2, input_truth], outputs=[mu_pred, sigma_pred])
    model.add_loss(loss(input_truth, mu_pred, sigma_pred))
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=None)
    
    return model


#------------------------------------------------------------------------------------------

##probabilistic loss functions
def loss_pred_std(y_true, y_pred, s_pred):
#     s_pred = tf.convert_to_tensor(s_pred) #x is y_true
    neg_log_unnormalized = 0.5 * tf.math.squared_difference(y_true / s_pred, y_pred / s_pred)
#      log_normalization = tf.constant(0.5 * np.log(2. * np.pi), shape=(448, 576,1), dtype=tf.float32)+ tf.math.log(s_pred)
#     print(tf.shape(tf.math.reduce_mean(-log_unnormalized + tf.math.log(s_pred))))
    return tf.math.reduce_mean(neg_log_unnormalized + tf.math.log(s_pred))

def Lgumbel(y_true, y_pred, gamma):
    x = y_true-y_pred # y_pred = mu
    xg = x/gamma
    mean = tf.math.reduce_mean(xg+tf.math.exp(-xg)) #+tf.math.log(gamma)) #this time no log gamma 

    #mean_gamma1 = tf.math.reduce_mean(x+tf.math.exp(-x))
#     mean = tf.math.reduce_mean(-tf.math.log(tf.math.exp(-xg)*tf.math.exp(-tf.math.exp(-xg))/gamma)) too numerically unstable
    return mean
#     return tf.math.reduce_mean((1-tf.math.exp(-e))*gamma*e, axis=-1) #maybe change axis to axis=-1

def Lgumbel_full(y_true, y_pred, gamma):
    x = y_true-y_pred
    xg=x/gamma
    mean = tf.math.reduce_mean(xg+tf.math.exp(-xg)+tf.math.log(gamma))
    return mean

def Lgumbel_exp(y_true, y_pred, gamma):
    x = y_true-y_pred
    xg=x/tf.exp(gamma)
    mean = tf.math.reduce_mean(xg+tf.math.exp(-xg))
    return mean
###TRAIN MODELS

####25KM
# c1 = checkpoint_callback1
# mod25 = mod25(Lgumbel, .00001) #prev LR at .00001
# hist25_gumbel = mod25.fit([xtrainMR, timetrain, tertrain, xtrainHR], callbacks=[c1], epochs=500, validation_split=0.1, verbose=2)
# # np.save('MRHR25kmodel_checkpoints/3channel_gumbel_test/training_history.npy', hist25_gumbel.history)
# np.save('MRHR25kmodel_checkpoints/3channel_gumbel_test/training_history_500epochs.npy', hist25_gumbel.history)

cL = cc_alllin
mod25 = mod25_alllin(Lgumbel, .00001)
hist25_gumbel = mod25.fit([xtrainMR, timetrain, tertrain, xtrainHR], callbacks=[cL], epochs=250, validation_split=0.1, verbose=2)
np.save('MRHR25kmodel_checkpoints/3channel_gumbel_linear/training_history.npy', hist25_gumbel.history)

#cLG = cc_log_gam
#mod25 = mod25_log_gamma(Lgumbel_exp, .00001)
#hist25_logG = mod25.fit([xtrainMR, timetrain, tertrain, xtrainHR], callbacks=[cLG], epochs=500, validation_split=0.1, verbose=2)
#np.save('MRHR25kmodel_checkpoints/3channel_gumbel_log_gamma/training_history.npy', hist25_logG.history)

# cLin = cc_lin_mu
# mod25 = mod25_alllin(Lgumbel_full, .00001)
# hist25_full_loss = mod25.fit([xtrainMR, timetrain, tertrain, xtrainHR], callbacks=[cLin], epochs=500, validation_split=0.1, verbose=2)
# np.save('MRHR25kmodel_checkpoints/3channel_gumbel_linear_full_loss/training_history.npy', hist25_full_loss.history)
#####25km




# c2 = checkpoint_callback2
# mod48 = mod48(Lgumbel, .00001)
# hist48_gumbel = mod48.fit([xtrainMR2, timetrain, tertrain, xtrainHR], callbacks=[c2], epochs=100, validation_split=0.1, verbose=2)
# np.save('MRHR48kmodel_checkpoints/3channel_gumbel/training_history.npy', hist48_gumbel)

# c3 = checkpoint_callback3
# mod100 = mod100(Lgumbel, .00001)
# hist100_gumbel = mod100.fit([xtrainLR, timetrain, tertrain, xtrainHR], callbacks=[c3], epochs=100, validation_split=0.1, verbose=2)
# np.save('LRHRmodel_checkpoints/3channel_gumbel/training_history.npy', hist100_gumbel)

#gaussians (save predictions to avoid reloading in notebook)
# c5 = checkpoint_callback5
# m25g = mod25(loss_pred_std, .00001)
# hist25_gauss = m25g.fit([xtrainMR, timetrain, tertrain, xtrainHR], callbacks=[c5], epochs=100, validation_split=0.1, verbose=2)
# np.save('MRHR25kmodel_checkpoints/3channel_gauss/training_history.npy', hist25_gauss.history)

# p5 = m25g.predict([xtestMR, timetest, tertest, xtestHR])
# np.save('MRHR25kmodel_checkpoints/3channel_gauss/gauss_preds.npy', p5)

# c6 = checkpoint_callback6
# m48g = mod48(loss_pred_std, .00001)
# hist48_gauss = m48g.fit([xtrainMR2, timetrain, tertrain, xtrainHR], callbacks=[c6], epochs=100, validation_split=0.1, verbose=2)
# np.save('MRHR48kmodel_checkpoints/3channel_gauss/training_history.npy', hist48_gauss.history)

# p6 = m48g.predict([xtestMR2, timetest, tertest, xtestHR])
# np.save('MRHR48kmodel_checkpoints/3channel_gauss/gauss_preds.npy', p6)

#c7 = checkpoint_callback7
#m100g = mod100(loss_pred_std, .00001)
#hist100_gauss = m100g.fit([xtrainLR, timetrain, tertrain, xtrainHR], callbacks=[c7], epochs=100, validation_split=0.1, verbose=2)
#np.save('LRHRmodel_checkpoints/3channel_gauss/training_history.npy', hist100_gauss.history)

# p7 = m100g.predict([xtestLR, timetest, tertest, xtestHR])
# np.save('LRHRmodel_checkpoints/3channel_gauss/gauss_preds.npy', p7)


