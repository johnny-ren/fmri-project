from keras.models import Sequential
from keras.layers import Dense, Dropout, RNN, GRU, AveragePooling1D
from keras import optimizers
from keras import utils
from keras.callbacks import TensorBoard

import numpy as np

#Variable to easily toggle between different data preprocessing methods
DATASET = 2

#Define hyperparameters and other details for the model
drop_out=0.30
r_drop_out=0.30
learning_rate=0.001
b1=0.9
b2=0.999
b_size=64
num_epochs=60

#Define TensorBoard callback. This allows for visualization of the data.
tbCallBack=TensorBoard(log_dir="logs/"+"ds"+str(DATASET)+"do"+str(drop_out)+"rdo"+str(r_drop_out)+
                       "lr"+str(learning_rate)+
                        "b1"+str(b1)+"b2"+str(b2)+
                        "bsize"+str(b_size)+
                        "numep"+str(num_epochs)
                       , write_graph=False)

#Load in the data

if DATASET == 0:
    loaded_data=np.load('/cfmriusr/data/ProcessedNumpyData/processed_numpy_data.npz')
elif DATASET == 1:
    loaded_data=np.load('/cfmriusr/data/ProcessedNumpyData/pre_shuffle_processed_numpy_data.npz')
elif DATASET ==2:
    loaded_data = np.load('/cfmriusr/data/ProcessedNumpyData/newest_unshuffled_correctdemean_processed_numpy_data.npz')

training_data=loaded_data['training_data']
training_labels=loaded_data['training_labels']

validation_data=loaded_data['validation_data']
validation_labels=loaded_data['validation_labels']

test_data=loaded_data['test_data']
test_labels=loaded_data['test_labels']

#Create model
model=Sequential()

#Average pooling layer to lower complexity of input to GRU
model.add(AveragePooling1D(input_shape=(100,379), pool_size=4))

#GRU with 256 dimensions
model.add(GRU(256, input_shape=(25,379), dropout=drop_out, recurrent_dropout=r_drop_out))

#Softmax layer which yields output
model.add(Dense(100, activation='softmax'))

#Define the optimizer used
adam_optimizer=optimizers.Adam(lr=learning_rate, beta_1=b1, beta_2=b2)

#Compile and and train the model
model.compile(optimizer=adam_optimizer,
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.fit(training_data, utils.to_categorical(training_labels, num_classes=100), batch_size=b_size, epochs=num_epochs,
          validation_data=(validation_data, utils.to_categorical(validation_labels, num_classes=100)), callbacks=[tbCallBack])

