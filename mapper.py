import numpy as np
import DataTransformer as dt

path = "C:\\Users\\rustahk\\Documents\\My Projects\\PerovClass\\"
training_data_abo = np.load(path+"datasets\\training_data.ABO Li v001.v001.npy")
training_labels_abo = np.load(path+"datasets\\training_labels.ABO Li v001.v001.npy")
test_data_abo = np.load(path+"datasets\\test_data.ABO Li v001.v001.npy")
test_labels_abo = np.load(path+"datasets\\test_labels.ABO Li v001.v001.npy")
training_data_abx = np.load(path+"datasets\\training_data.ABX Li v001.v001.npy")
training_labels_abx = np.load(path+"datasets\\training_labels.ABX Li v001.v001.npy")
test_data_abx = np.load(path + "datasets\\test_data.ABX Li v001.v001.npy")
test_labels_abx = np.load(path+"datasets\\test_labels.ABX Li v001.v001.npy")
training_data = np.concatenate((training_data_abo, training_data_abx))
training_labels = np.concatenate((training_labels_abo, training_labels_abx))
test_data = np.concatenate((test_data_abo, test_data_abx))
test_labels = np.concatenate((test_labels_abo, test_labels_abx))
data = np.concatenate((training_data, test_data))
labels = np.concatenate((training_labels, test_labels))
data = np.delete(data, 0, 1)  # remove id
data = dt.cast_to_properties(data)
# remove features
data = np.delete(data, [1, 3, 4], 2)

for i in range(len(labels)):
    print('***')
    print(data[i,0])