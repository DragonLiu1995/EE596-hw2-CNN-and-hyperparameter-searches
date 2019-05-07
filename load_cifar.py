import pickle
import numpy as np
import math
#Step 1: define a function to load traing batch data from directory
"""
Args:
folder_path: the directory contains data files
batch_id: training batch id (1,2,3,4,5)
Return:
features: numpy array that has shape (10000,3072)
labels: a list that has length 10000
"""
def load_training_batch(folder_path,batch_id):
    ###load batch using pickle###
    with open(folder_path+"/data_batch_"+str(batch_id), 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
    ###fetch features using the key ['data']###
        features = datadict['data']
    ###fetch labels using the key ['labels']###
        labels = datadict['labels']
    return features,labels

#Step 2: define a function to load testing data from directory

"""
Args:
folder_path: the directory contains data files
Return:
features: numpy array that has shape (10000,3072)
labels: a list that has length 10000
"""
def load_testing_batch(folder_path):
    ###load batch using pickle###
    with open(folder_path+"/test_batch", 'rb') as f:
         datadict = pickle.load(f, encoding='latin1')
    ###fetch features using the key ['data']###
         features = datadict['data']
    ###fetch labels using the key ['labels']###
         labels = datadict['labels']
    return features,labels

#Step 3: define a function that returns a list that contains label names (order is matter)
"""
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""
def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Step 4: define a function that reshapes the features to have shape (10000, 32, 32, 3)

"""
Args:
features: a numpy array with shape (10000, 3072)
Return:
features: a numpy array with shape (10000,32,32,3)
"""
def features_reshape(features):
    features = features.reshape((10000,32,32,3))
    return features

#Step 5 (Optional): A function to display the stats of specific batch data.
"""
Args:
folder_path: directory that contains data files
batch_id: the specific number of batch you want to explore.
data_id: the specific number of data example you want to visualize
Return:
    None
Descrption: 
    1)You can print out the number of images for every class. 
    2)Visualize the image
    3)Print out the minimum and maximum values of pixel 
"""
def display_data_stat(folder_path,batch_id,data_id):
    pass

#Step 6: define a function that does min-max normalization on input
"""
Args:
x: features, a numpy array
Return:
x: normalized features
"""
def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x


#Step 7: define a function that does one hot encoding on input
"""
Args:
x: a list of labels
Return:
a numpy array that has shape (len(x), # of classes)
"""
def one_hot_encoding(x):
    X = np.zeros((len(x), 10))
    for i in range(len(x)):
           X[i][x[i]] = 1
    X.astype(np.float32)
    return X

#Step 8: define a function that perform normalization, one-hot encoding and save data using pickle
"""
Args:
features: numpy array
labels: a list of labels
filename: the file you want to save the preprocessed data
"""
def preprocess_and_save(features,labels,filename):
    features = normalize(features)
    features = features.reshape((features.shape[0],32,32,3))
    labels = one_hot_encoding(labels)
    pickle.dump((features, labels), open(filename, 'wb'))

#Step 9:define a function that preprocesss all training batch data and test data. 
#Use 10% of your total training data as your validation set
#In the end you should have 5 preprocessed training data, 1 preprocessed validation data and 1 preprocessed test data

"""
Args:
folder_path: the directory contains your data files
"""
def preprocess_data(folder_path):
    batches = 5
    val_features = []
    val_labels = []
    for batch_i in range(1, batches + 1):
        features, labels = load_training_batch(folder_path, batch_i)
        idx = int(len(features) * 0.1)
        preprocess_and_save(features[:-idx], labels[:-idx], 'preprocess_batch_' + str(batch_i) + '.p')
        val_features.extend(features[-idx:])
        val_labels.extend(labels[-idx:])
        
    # preprocess validation dataset
    preprocess_and_save(np.array(val_features), np.array(val_labels), 'preprocess_validation.p')


    # preprocess the testing data
    test_features, test_labels = load_testing_batch(folder_path)
    preprocess_and_save(np.array(test_features), np.array(test_labels), 'preprocess_test.p')

#Step 10: define a function to yield mini_batch
"""
Args:
features: features for one batch
labels: labels for one batch
mini_batch_size: the mini-batch size you want to use.
Hint: Use "yield" to generate mini-batch features and labels
"""
def mini_batch(features,labels,mini_batch_size):
    l = features.shape[0]
    for ndx in range(0, l, mini_batch_size):
        low = ndx
        high = min((ndx + mini_batch_size), l)
        yield features[low:high,:], labels[low:high,:]

#Step 11: define a function to load preprocessed training batch, the function will call the mini_batch() function
"""
Args:
batch_id: the specific training batch you want to load
mini_batch_size: the number of examples you want to process for one update
Return:
    mini_batch(features,labels, mini_batch_size)
"""
def load_preprocessed_training_batch(batch_id,mini_batch_size):
    file_name = 'preprocess_batch_' + str(batch_id) + '.p'
    data = pickle.load(open(file_name, 'rb'))
    features, labels = data[0], data[1]
    return mini_batch(features,labels,mini_batch_size)

#Step 12: load preprocessed validation batch
def load_preprocessed_validation_batch():
    file_name = 'preprocess_validation.p'
    data = pickle.load(open(file_name, 'rb'))
    features,labels = data[0], data[1]
    return features,labels

#Step 13: load preprocessed test batch
def load_preprocessed_test_batch(test_mini_batch_size):
    file_name = 'preprocess_test.p'
    data = pickle.load(open(file_name, 'rb'))
    features,labels = data[0], data[1]
    return mini_batch(features,labels,test_mini_batch_size)

