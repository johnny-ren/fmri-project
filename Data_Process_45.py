import h5py
import numpy as np

#Load in the data
f = h5py.File('/cfmriusr/data/Processed100/S500_alldata_with_subcort_normalized.mat','r')
data = f.get('alldata')
data = np.array(data)

#Dimensions of the data are 100,4,1200,379. It is a four dimensional array.
print(data.shape)

#Remove one dimension. Instead of having four scans (in second dimension) for each individual (in first dimension),
#have all scans laid out on one dimension

data=np.concatenate((data[:, 0, :, :], data[:, 1, :, :], data[:, 2, :, :], data[:, 3, :, :]), axis=0)

#Check the shape of the data
print(data.shape)

#Cut each scan into pieces 100 time frames long
temp_data=np.zeros((18000, 100, 379))
for x in range(data.shape[0]):
    for y in range(45):
        data_portion=data[x, :, :]
        #print(temp_data[12*x+y,:,:].shape)
        #print(train_data_portion.shape)
        temp_data[45*x+y,:,:]=data_portion[y*25: y*25+100, :]


data=temp_data
print(data.shape)

#Create labels for the data
labels=np.zeros(18000)
for x in range(400):
    labels[x*45:x*45+45]=x%100

print(labels)

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
            std=np.std(sliver)
            three_d_matrix[x, :, y]=three_d_matrix[x, :, y]-avg
            three_d_matrix[x, :, y] = three_d_matrix[x, :, y]/std

    return three_d_matrix

data=demean_and_normalize(data)
print('Finished demeaning and normalizing')

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


def shuffle_together(a,b):
    assert a.shape[0]==b.shape[0]
    p=np.random.permutation(a.shape[0])
    return a[p], b[p]


#Separate training, validation, and test data
training_data=data[0:9000, :, :]
print(training_data.shape)
validation_data=data[9000:13500, :, :]
print(validation_data.shape)
test_data=data[13500:18000, :, :]
print(test_data.shape)

#Separate training, validation, and test data
training_labels=labels[0:9000]
print(training_labels.shape)
validation_labels=labels[9000:13500]
print(validation_labels.shape)
test_labels=labels[13500:18000]
print(test_labels.shape)

np.savez('/cfmriusr/data/ProcessedNumpyData/newest_unshuffled_correctdemean_processed_numpy_data', training_data=training_data,
         training_labels=training_labels,
         validation_data=validation_data,
         validation_labels=validation_labels,
         test_data=test_data,
         test_labels=test_labels)

# training_data, training_labels=shuffle_together(training_data, training_labels)
# print(training_labels)
#
# validation_data, validation_labels=shuffle_together(validation_data, validation_labels)
# print(validation_labels)
#
# test_data, test_labels=shuffle_together(test_data, test_labels)
# print(test_labels)
#
# np.savez('/cfmriusr/data/ProcessedNumpyData/processed_numpy_data', training_data=training_data,
#          training_labels=training_labels,
#          validation_data=validation_data,
#          validation_labels=validation_labels,
#          test_data=test_data,
#          test_labels=test_labels)