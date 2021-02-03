from tensorflow import keras
import DataTransformer as dt
import numpy as np
import mendeleev as md

def contrast(model, data, labels):
    model.evaluate(data, labels)
    predictions = model.predict(data)
    predictions[predictions <= 0.5] = 0.0
    predictions[predictions > 0.5] = 1.0
    for i in range(len(labels)):
        if (predictions[i] != labels[i]):
            temp = data[i] * 100
            name = ""
            for j in temp[:, 0]:
            # for j in temp:
                el = md.element(int(round(j)))
                name += el.symbol
            print(name, "prediction: ",predictions[i] )


path = "C:\\Users\\rustahk\\Documents\\My Projects\\PerovClass\\"

# load data #
training_data_abo = np.load(path+"datasets\\training_data.ABO Li v002.v001.npy")
training_labels_abo = np.load(path+"datasets\\training_labels.ABO Li v002.v001.npy")
test_data_abo = np.load(path+"datasets\\test_data.ABO Li v002.v001.npy")
test_labels_abo = np.load(path+"datasets\\test_labels.ABO Li v002.v001.npy")
training_data_abx = np.load(path+"datasets\\training_data.ABX Li v002.v001.npy")
training_labels_abx = np.load(path+"datasets\\training_labels.ABX Li v002.v001.npy")
test_data_abx = np.load(path + "datasets\\test_data.ABX Li v002.v001.npy")
test_labels_abx = np.load(path+"datasets\\test_labels.ABX Li v002.v001.npy")
training_data = np.concatenate((training_data_abo, training_data_abx))
training_labels = np.concatenate((training_labels_abo, training_labels_abx))
test_data = np.concatenate((test_data_abo, test_data_abx))
test_labels = np.concatenate((test_labels_abo, test_labels_abx))
# remove id
training_data = np.delete(training_data, 0, 1)
test_data = np.delete(test_data, 0, 1)
# cast to features
training_data = dt.cast_to_properties(training_data)
test_data = dt.cast_to_properties(test_data)
# normalization
training_data = dt.normalize(training_data)
test_data = dt.normalize(test_data)
# remove features
training_data = np.delete(training_data, [1], 2)
test_data = np.delete(test_data, [1], 2)

#
data = np.concatenate((training_data, test_data))
labels = np.concatenate((training_labels, test_labels))

model = keras.models.load_model('best\\M_abx3+abo3_n_ri_xsi_rc_tr99.25ts94.78')
model.summary()


print(np.shape(training_data))
print("training")
contrast(model, training_data, training_labels)
print("test")
contrast(model, test_data, test_labels)
print("total")
contrast(model, data, labels)
