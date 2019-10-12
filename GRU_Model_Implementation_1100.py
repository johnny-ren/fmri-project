from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, AveragePooling1D
from keras import optimizers
from keras import utils
from keras.callbacks import TensorBoard, ModelCheckpoint

import h5py

import numpy as np

def demean_and_normalize(three_d_matrix):
    for x in range(three_d_matrix.shape[0]):
        for y in range(three_d_matrix.shape[2]):
            sliver=three_d_matrix[x, :, y]
            avg=np.average(sliver)
            std=np.std(sliver)
            three_d_matrix[x, :, y] = three_d_matrix[x, :, y]-avg
            three_d_matrix[x, :, y] = three_d_matrix[x, :, y]/std

    return three_d_matrix


def generate_arrays_from_file():
    f = h5py.File('/cfmriusr/data/Processed100/S500_alldata_with_subcort_normalized.mat', 'r')
    data = f.get('alldata')
    #Convert to numpy array
    data = np.array(data)
    data = np.concatenate((data[:, 0, :, :], data[:, 1, :, :]), axis=0)

    indices=np.zeros((220000,2), dtype=int)

    keep_track=0
    for x in range(200):
        for y in range(1100):
            indices[keep_track, :]=[x, y]
            keep_track+=1

    #print(indices)

    labels=np.zeros((220000), dtype=int)

    for x in range(200):
        labels[x*1100:x*1100+1100] = x % 100

    p=np.random.permutation(indices.shape[0])
    print(p)
    labels=labels[p]
    indices=indices[p]
    index=0
    while True:
        batch=np.zeros((64,100,379))
        batch_labels=np.zeros((64))
        for i in range(64):
            ryan_howard=indices[index, :]

            sliver=data[ryan_howard[0], ryan_howard[1]: ryan_howard[1]+100, :]

            batch[i, :, :] = sliver
            batch_labels[i]=labels[index]
            index += 1
            if index >= 220000:
                index=0
                p = np.random.permutation(indices.shape[0])

                labels = labels[p]
                indices = indices[p]
                print(p)
        batch=demean_and_normalize(batch)
        yield batch, utils.to_categorical(batch_labels, num_classes=100)

#The validation generator is provided here
def generate_arrays_from_file_val():
    f = h5py.File('/cfmriusr/data/Processed100/S500_alldata_with_subcort_normalized.mat', 'r')
    data = f.get('alldata')
    #Convert to numpy array
    data = np.array(data)
    data = data[:, 3, :, :]

    indices=np.zeros((110000,2), dtype=int)

    keep_track=0
    for x in range(100):
        for y in range(1100):
            indices[keep_track, :]=[x, y]
            keep_track+=1

    #print(indices)

    labels=np.zeros((110000), dtype=int)

    for x in range(100):
        labels[x*1100:x*1100+1100] = x % 100

    p=np.random.permutation(indices.shape[0])
    print(p)
    labels=labels[p]
    indices=indices[p]
    index=0
    while True:
        batch=np.zeros((64,100,379))
        batch_labels=np.zeros((64))
        for i in range(64):
            ryan_howard=indices[index, :]

            sliver=data[ryan_howard[0], ryan_howard[1]: ryan_howard[1]+100, :]

            batch[i, :, :] = sliver
            batch_labels[i]=labels[index]
            index += 1
            if index >= 110000:
                index=0
                p = np.random.permutation(indices.shape[0])

                labels = labels[p]
                indices = indices[p]
                print(p)
        batch=demean_and_normalize(batch)
        yield batch, utils.to_categorical(batch_labels, num_classes=100)

#Define hyperparameters and other details for the model
drop_out=0.45
r_drop_out=0.45
learning_rate=0.001
b1=0.9
b2=0.999
b_size=64
num_epochs=2
steps_per_epoch=1500

DATASET=4

tbCallBack=TensorBoard(log_dir="logs/"+"spe"+str(steps_per_epoch)+"ds"+str(DATASET)+"do"+str(drop_out)+"rdo"+str(r_drop_out)+
                       "lr"+str(learning_rate)+
                        "b1"+str(b1)+"b2"+str(b2)+
                        "bsize"+str(b_size)+
                        "numep"+str(num_epochs)
                       , write_graph=False)

checkCallBack=ModelCheckpoint('weights.best.hdf5', monitor='val_acc', save_best_only=True)

#The data loaded below can be used if one does not want to use a generator for the validation or test data
loaded_data = np.load('/cfmriusr/data/ProcessedNumpyData/validation_data.npz')

validation_data=loaded_data['validation_data']
validation_labels=loaded_data['validation_labels']

# test_data=loaded_data['test_data']
# test_labels=loaded_data['test_labels']

model=Sequential()

model.add(AveragePooling1D(input_shape=(100,379), pool_size=4))

model.add(Dropout(0.25))

model.add(GRU(256, dropout=drop_out, recurrent_dropout=r_drop_out))

model.add(Dense(100, activation='softmax'))

adam_optimizer=optimizers.Adam(lr=learning_rate, beta_1=b1, beta_2=b2)

model.compile(optimizer=adam_optimizer,
             loss='categorical_crossentropy',
             metrics=['accuracy'])

history=model.fit_generator(generate_arrays_from_file(), steps_per_epoch=steps_per_epoch,
           validation_data=(validation_data, utils.to_categorical(validation_labels, num_classes=100)), epochs=num_epochs,
                    callbacks=[tbCallBack, checkCallBack])

#Use the below fit generator if you would also like to use a generator for the validation data. Note: The validation process
#takes much longer for this version.
# history=model.fit_generator(generate_arrays_from_file(), steps_per_epoch=steps_per_epoch,
#           validation_data=generate_arrays_from_file_val(), validation_steps=1718, epochs=num_epochs,
#                     callbacks=[tbCallBack, checkCallBack])
