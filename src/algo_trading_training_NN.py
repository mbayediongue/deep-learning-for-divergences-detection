#!/usr/bin/env python
# coding: utf-8

# ##                   PRICE: Machine learning pour le trading algorithmique
# 
# 18/05/2022
# 
# Ce document contient le code nécesaire à l'entrainement du réseau de neurones (RN). Le RN permet de prédire la qualité des divergences détectées dans l'algorithme développé lors du projet PRICE 2021. Cette qualité peut être interprétée comme la hauteur de chute (ou la montée) attendue dans les cours boursiers. Par conséquent il permet de déterminer le montant à investir (ou retirer) et le temps au bout duquel il est conseillé de retirer sa mise (stop lose/ stop loss).
# 

# In[155]:


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.layers import Input, Flatten, Dense, Concatenate, LSTM, Dropout
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from numpy.random import seed
from sklearn.metrics import mean_squared_error 
import math
import os

# In[178]:


# to be modified (the repository should be the one hosting the data)
#get_ipython().run_line_magic('cd', '"C:\\Espace étude\\EMSE 2A\\PRICE\\algorithmic_trading-main\\algorithmic_trading-main\\Trading_project\\algo_parallele"')


# ### 1. Loading data

# In[180]:


with open("dataTot.out", 'r') as file:
    data_= file.readlines()
    
data= np.loadtxt(data_, delimiter=',')
print(data.shape)
data=pd.DataFrame(data)
data.dropna(inplace=True) 
print(data.shape)
data=np.array(data)
#print(data)
#data=np.unique(data, axis=0)
X, Y= np.delete(data, 5, axis=1), data[:,5]


# In[182]:


Y=Y.reshape(-1, 1)
print(X.shape)
print(Y.shape)


# ### 2. Splitting data into training, test and validation set

# In[183]:


train_size=0.8 # size of the training set
(X_train, X_rem, Y_train, Y_rem) = train_test_split(X, Y, train_size=train_size, random_state=None)

# Validation and test set
test_size=0.5 # test set size in the remaining data
(X_test, X_val, Y_test, Y_val) = train_test_split(X_rem, Y_rem, test_size=test_size, random_state=None)


# ### 3. Normalization of the data

# In[186]:


# Inputs
scalerX = StandardScaler().fit(X_train)
X_train = scalerX.transform(X_train)
X_val = scalerX.transform(X_val)
X_test = scalerX.transform(X_test)

# Outputs
scalerY = StandardScaler().fit(Y_train)
Y_train_ori=Y_train.copy()
Y_train = scalerY.transform(Y_train)
#Y_val = scalerY.transform(Y_val)
#Y_test = scalerY.transform(Y_test)


# ### 4. Pretreament for the lstm network entry and the first dense network entry

# In[188]:


X_train_lstm_temp, X_train2= X_train[:,5:], X_train[:, 0:5]
X_val_lstm_temp, X_val2= X_val[:,5:], X_val[:, 0:5]
X_test_lstm_temp, X_test2= X_test[:,5:], X_test[:, 0:5]

X_train_lstm= np.empty((X_train_lstm_temp.shape[0], X_train_lstm_temp.shape[1]//2, 2))
X_train_lstm[:,:,0]=X_train_lstm_temp[:,0:X_train_lstm_temp.shape[1]//2]
X_train_lstm[:,:,1]=X_train_lstm_temp[:,X_train_lstm_temp.shape[1]//2 :]

X_val_lstm= np.empty((X_val_lstm_temp.shape[0], X_val_lstm_temp.shape[1]//2, 2))
X_val_lstm[:,:,0]=X_val_lstm_temp[:,0:X_val_lstm_temp.shape[1]//2]
X_val_lstm[:,:,1]=X_val_lstm_temp[:,X_val_lstm_temp.shape[1]//2 :]

X_test_lstm= np.empty((X_test_lstm_temp.shape[0], X_test_lstm_temp.shape[1]//2, 2))
X_test_lstm[:,:,0]=X_test_lstm_temp[:,0:X_test_lstm_temp.shape[1]//2]
X_test_lstm[:,:,1]=X_test_lstm_temp[:,X_test_lstm_temp.shape[1]//2 :]

#X_train_lstm=np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1]//2, 2 ))
print("X_train_lstm shape: ", X_train_lstm.shape)
print("X_validattion_lstm shape: ", X_val_lstm.shape)
print("X_test_lstm shape: ", X_test_lstm.shape)

print("X_train2 shape: ", X_train2.shape)
print("Y_train shape: ", Y_train.shape)


"""
# ### 5.1 First neural network LSTM 
# 
# Our first model is an LSTM neural network. So we use ONLY temporal data and do not consider other 5 predictors (slop of the stock and the RSI, duration of the divergence ...)

# In[ ]:


# sauvegarde des poids au cours de l'apprentissage...
checkpoint_path = "training_LSTM_only/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size=32
# Create a callback that saves the model's weights every 4 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=4*batch_size)



# In[189]:


seed(1234) # Pour la reproductibilité
tf.random.set_seed(1234)
model_lstm = Sequential()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)
model_lstm.add(LSTM(16, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]) ))
#model.add(LSTM(5,input_shape=(1, look_back), activation='relu'))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(X_train_lstm, Y_train, epochs=50, batch_size=5, verbose=1, callbacks=[es, cp_callback])


# Save the weights using the `checkpoint_path` format
model_lstm.save_weights(checkpoint_path.format(epoch=0))
model_lstm_save
"""
# ## 5.2 LSTM combined with fully-connected layers 
# This second model combined an LSTM part (fo temporal data) and a classic network part (for other variables: slops of the stock market and the RSI, duration pf the divergence ...). Theses to subnetwork are combined at the end by a fully connected layer that predicts the Y value.

# In[ ]:


# sauvegarde des poids au cours de l'apprentissage...
checkpoint_path = "training_LSTM_DENSE/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size=32
# Create a callback that saves the model's weights every 4 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=4*batch_size)



# In[ ]:


nbVar=5 # number of other variables distinct from the RSI and Cours
context_lenght=30

# feature extraction from temporal data (cours+rsi)
#        X_train_lstm.shape[1]= time_step
temporal_series_input = Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), name='temporal_input')
lstm1=LSTM(32, return_sequences = True)(temporal_series_input)
dropout1 = Dropout(0.3)(lstm1, training=True)
lstm2=LSTM(16, return_sequences = True)(dropout1)
dropout2 = Dropout(0.3)(lstm2, training=True)
dense1_1=Dense( 8,activation = 'relu'  )(dropout2)
#lstm3=LSTM(8, activation = 'relu')(dropout2)
flat_1 = Flatten()(dense1_1)

# feature extraction from other variables ('angle1', 'angle2', 'longueur', 'type divergence')
other_inputs  = Input(shape=(X_train2.shape[1],), name='other_variables')
dense2_1=Dense( 16,activation = 'relu'  )(other_inputs)
dropout2_1 = Dropout(0.3)(dense2_1, training=True)
dense2_2=Dense( 8,activation = 'relu'  )(dropout2_1)
#dropout2_2 = Dropout(0.3)(dense2_2, training=True)
#dense2_3=Dense( 8,activation = 'relu'  )(dropout2_2)
flat_2 = Flatten()(dense2_2)

# concatenate both feature layers and define output layer after some dense layers
concat = Concatenate()([flat_1,flat_2])
dense1 = Dense(16, activation = 'relu')(concat)
dense2 = Dense(8, activation = 'relu')(dense1)
output = Dense(1)(dense2)
#output = Dense(10, activation = 'softmax')(dense3)

# create model with two inputs
model_lstm_dense= Model(inputs=[temporal_series_input, other_inputs], outputs=output)
model_lstm_dense.summary()

# Save the weights using the `checkpoint_path` format
model_lstm_dense.save_weights(checkpoint_path.format(epoch=0))

# ### 

# In[177]:


#tf.random.set_see
step=100
initial_learning_rate=1e-3
step=tf.Variable(0, trainable=False)
print(step)
#tf.compat.v1.train.exponential_decay(starter_learning_rate,global_step, 100000, 0.96, staircase=True)
learning_rate = tf.compat.v1.train.exponential_decay(initial_learning_rate, step, 1000, 0.999, staircase=True)
opt = tf.keras.optimizers.Adam(learning_rate)
model_lstm_dense.compile(loss='mean_squared_error', optimizer=opt)
# to avoid overfitting
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True) 

history = model_lstm_dense.fit( [X_train_lstm, X_train2],
                    Y_train,
                    epochs          = 500,
                    batch_size      = batch_size,
                    verbose         = 1,
                    validation_data = ([X_val_lstm, X_val2], Y_val),
                    callbacks=[es, cp_callback]
                   )


# plt.plot(history.history['mse'] )
# plt.plot(history.history['val_mse'])

# ### 6. Testing with validation set and training set for parameters optimization

# In[ ]:

model=model_lstm_dense
# On fait les prédictions
# .... Sur les données d'apprentissage
Y_train_predict = model.predict([X_train_lstm, X_train2])  # hauteur de chute prédite sur les données d'apprentissage
Y_train_predict= scalerY.inverse_transform(Y_train_predict) # repassage à l'échelle
#trainY_ = scaler.inverse_transform([trainY])

np.savetxt('Y_train_predict.out', Y_train_predict, delimiter=',')   # X is an array
np.savetxt('y_train_real.out', Y_train_ori, delimiter=',')   # X is an array


# ... Sur la validation set 
Y_val_predict = model.predict([X_val_lstm, X_val2])
Y_val_predict = scalerY.inverse_transform(Y_val_predict)

np.savetxt('Y_val_predict.out', Y_train_predict, delimiter=',')   # X is an array
np.savetxt('y_val_real.out', Y_val, delimiter=',')   # X is an array

#testY_= scaler.inverse_transform([testY])

# Compute root mean squared error
trainScore = math.sqrt(mean_squared_error(Y_train_ori[0], Y_train_predict[:,0]))
print('Train Score: %.5f RMSE' % (trainScore))
valScore = math.sqrt(mean_squared_error(Y_val[0], Y_val_predict[:,0]))
print('Test Score: %.5f RMSE' % (valScore))

# ### 7. Test set

# In[ ]:


# ... Sur les données de  test 
Y_test_predict = model.predict([X_test_lstm, X_test2])
Y_test_predict = scalerY.inverse_transform(Y_test_predict)
#testY_= scaler.inverse_transform([testY])

np.savetxt('Y_test_predict.out', Y_test_predict, delimiter=',')   # X is an array
np.savetxt('y_test_real.out', Y_test, delimiter=',')   # X is an array
# Compute root mean squared error
testScore = math.sqrt(mean_squared_error(Y_test[0], Y_test_predict[:,0]))
print('Test Score: %.5f RMSE' % (testScore))


# In[ ]:


plt.plot(Y)

