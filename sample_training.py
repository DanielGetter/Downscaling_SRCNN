from models import model1C25
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

#include sin_95 in data processing .py file
from data_processing import sin_95

#custom callback
checkpoint_callback1 = tf.keras.callbacks.ModelCheckpoint(filepath='MRHR25kmodel_checkpoints/1channel/model_epoch:{epoch:02d}.h5', 
	save_best_only=False, 
	save_weights_only=False, 
	monitor='val_loss', 
	save_freq=990, 
	verbose=1)

d = np.load('split_data_US.npz')
xtrainHR, xtrainMR, xtrainMR2, xtrainMR12, xtrainLR, tertrain, timetrain, xtestHR, xtestMR, xtestMR2, xtestMR12, xtestLR, tertest, timetest = d['name1'], d['name2'], d['name3'], d['name4'], d['name5'], d['name6'], d['name7'], d['name8'], d['name9'], d['name10'], d['name11'], d['name12'], d['name13'], d['name14']
#transform data#
xtrainHR, xtrainMR, xtrainLR, tertrain, xtestHR, xtestMR, xtestLR, tertest = np.sqrt(xtrainHR), np.sqrt(xtrainMR), np.sqrt(xtrainLR), np.sqrt(tertrain+1), np.sqrt(xtestHR), np.sqrt(xtestMR), np.sqrt(xtestLR), np.sqrt(tertest+1) 
xtrainMR2, xtestMR2 = np.sqrt(xtrainMR2), np.sqrt(xtestMR2)
xtrainMR12, xtestMR12 = np.sqrt(xtrainMR12), np.sqrt(xtestMR12)
timetrain, timetest = sin_95(timetrain), sin_95(timetest) #

#compile and train
saver1C = checkpoint_callback1
mod1C = model1C25()
mod1C.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.0001), loss=tf.keras.losses.MeanSquaredError())
hist1C = mod1C.fit(xtrainMR, xtrainHR, callbacks=[saver1C], epochs=500, validation_data=(xtestMR, xtestHR), verbose=2)
np.save('MRHR25kmodel_checkpoints/1channel/training_history.npy', hist1C.history)



#validation on OOS data
dES = whatever



