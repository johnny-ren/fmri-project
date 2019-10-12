import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

#This is the first python script written for this project.

#Load in the data
f = h5py.File('/cfmriusr/data/Processed100/S500_alldata_with_subcort_normalized.mat','r')
data = f.get('alldata')
#Convert to numpy array
data = np.array(data)

#Dimensions of the data are 100,4,1200,379. It is a four dimensional array.
print(data.shape)

#Isolate training data
train_data = np.concatenate((data[:, 0, :, :], data[:, 1, :, :]), axis=0)

test_matrix=np.zeros((64,100,379))

loaded_data=np.load('/cfmriusr/data/TestNumpyData/test.npz')
test=loaded_data['test_numpy']
print(test)

print(train_data.shape)

#Cut each scan into pieces 100 time frames long
temp_data=np.zeros((2400, 100, 379))
for x in range(train_data.shape[0]):
    for y in range(12):
        train_data_portion=train_data[x, :, :]
        #print(temp_data[12*x+y,:,:].shape)
        #print(train_data_portion.shape)
        temp_data[12*x+y,:,:]=train_data_portion[y*100: y*100+100, :]

train_data=temp_data
print(train_data.shape)

#Create labels for the data
training_labels=np.zeros(2400)
for x in range(200):
    training_labels[x*12:x*12+12]=x%100

print(training_labels)

# def createTrain(three_d_matrix, step_size):
#     temp=np.zeros((three_d_matrix.size[0]*three_d_matrix/step_size))
#     for x in range(three_d_matrix.shape[1]-step_size):
#


#Demean and Normalize the data
#The input should be a three dimensional matrix
#Number of scans x Number of Time Frames (Per Scan) x Number of ROIs
def demean_and_normalize(three_d_matrix):
    for x in range(three_d_matrix.shape[0]):
        for y in range(three_d_matrix.shape[2]):
            sliver=three_d_matrix[x, :, y]
            avg=np.average(sliver)
            three_d_matrix

    return three_d_matrix


model=Sequential()

model.add(Dense(100, activation='softmax'))

adam_optimizer=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer=adam_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
