import h5py
import numpy as np

#Load in the data
f = h5py.File('/cfmriusr/data/Processed100/S500_alldata_with_subcort_normalized.mat','r')
data = f.get('alldata')

#Convert to numpy array
data = np.array(data)

#Dimensions of the data are 100,4,1200,379. It is a four dimensional array.
print(data.shape)

#Remove one dimension. Instead of having four scans (in second dimension) for each individual (in first dimension),
#have all scans laid out on one dimension
data=np.concatenate((data[:, 0, :, :], data[:, 1, :, :], data[:, 2, :, :], data[:, 3, :, :]), axis=0)

#Check the shape of the data. Now it is 400, 1200, 379
print(data.shape)

#Cut each scan into pieces 100 time frames long. Create temporary matrix to store the data.
temp_data=np.zeros((4800, 100, 379))

#Get 12 chunks per scan
for x in range(data.shape[0]):
    for y in range(12):
        data_portion=data[x, :, :]
        temp_data[12*x+y,:,:]=data_portion[y*100: y*100+100, :]

#The data is now of size 4800, 100, 379
data=temp_data
print(data.shape)

#Create labels for the data
labels=np.zeros(4800)
for x in range(400):
    labels[x*12:x*12+12]=x%100

#Demean and Normalize Function
#The input should be a three dimensional matrix
#Number of scans x Number of Time Frames (Per Scan) x Number of ROIs
def demean_and_normalize(three_d_matrix):
    for x in range(three_d_matrix.shape[0]):
        for y in range(three_d_matrix.shape[2]):
            sliver=three_d_matrix[x, :, y]
            avg=np.average(sliver)
            three_d_matrix[x, :, y]=three_d_matrix[x, :, y]-avg
            std=np.std(sliver)
            three_d_matrix[x, :, y] = three_d_matrix[x, :, y]/std

    return three_d_matrix

#Demean and Normalize
data=demean_and_normalize(data)
print('Finished demeaning and normalizing')

#Note about below section: The every_four function was written for an earlier implementation of the model.
#In later models, it is replaced by the average pooling layer provided by keras

#Average every four time points into one. Effectively each time course which is 100 TF long becomes 25 TF long
def every_four(three_d_matrix):
    ryan_howard=np.zeros((three_d_matrix.shape[0], 25, three_d_matrix.shape[2]))
    for x in range(three_d_matrix.shape[0]):
        for y in range(three_d_matrix.shape[2]):
            sliver_25=np.zeros(25)
            sliver=three_d_matrix[x, :, y]
            for z in range(25):
                sliver_25[z]=np.average(sliver[z*4:z*4+4])
            ryan_howard[x,:,y]=sliver_25

    return ryan_howard

#data=every_four(data)
print('Finished averaging every four TFs')

#A shuffle function that shuffles labels and training data together
#Can be replaced by keras shuffle functionality in model fit
def shuffle_together(a,b):
    assert a.shape[0]==b.shape[0]
    p=np.random.permutation(a.shape[0])
    return a[p], b[p]


#Separate training, validation, and test data
training_data=data[0:2400, :, :]
print(training_data.shape)
validation_data=data[2400:3600, :, :]
print(validation_data.shape)
test_data=data[3600:4800, :, :]
print(test_data.shape)

#Separate training, validation, and test labels
training_labels=labels[0:2400]
print(training_labels.shape)
validation_labels=labels[2400:3600]
print(validation_labels.shape)
test_labels=labels[3600:4800]
print(test_labels.shape)

#Save the unshuffled data

# np.savez('/cfmriusr/data/ProcessedNumpyData/pre_shuffle_processed_numpy_data', training_data=training_data,
#          training_labels=training_labels,
#          validation_data=validation_data,
#          validation_labels=validation_labels,
#          test_data=test_data,
#          test_labels=test_labels)

training_data, training_labels=shuffle_together(training_data, training_labels)
print(training_labels)

validation_data, validation_labels=shuffle_together(validation_data, validation_labels)
print(validation_labels)

test_data, test_labels=shuffle_together(test_data, test_labels)
print(test_labels)

#Save the shuffled data

# np.savez('/cfmriusr/data/ProcessedNumpyData/processed_numpy_data', training_data=training_data,
#          training_labels=training_labels,
#          validation_data=validation_data,
#          validation_labels=validation_labels,
#          test_data=test_data,
#          test_labels=test_labels)