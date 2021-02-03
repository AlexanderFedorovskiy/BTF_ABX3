import numpy as np
import networks.seqnet as sq
from tensorflow import keras
import matplotlib.pyplot as plt


def plot_history(histories, key='accuracy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()
  plt.xlim([0,max(history.epoch)])
  plt.show()

# load data
path = "C:\\Users\\rustahk\\Documents\\My Projects\\PerovClass\\"
training_data = np.load(path+"datasets\\training_data.ABX Li v002.v001.npy")
training_labels = np.load(path+"datasets\\training_labels.ABX Li v002.v001.npy")
test_data = np.load(path+"datasets\\test_data.ABX Li v002.v001.npy")
test_labels = np.load(path+"datasets\\test_labels.ABX Li v002.v001.npy")
# remove id, group
training_data = np.delete(training_data, [0, 4], 1)
test_data = np.delete(test_data, [0, 4], 1)
# normalization
training_data = training_data/100
test_data = test_data/100

model = sq.build_model(660, 3, optimizer=keras.optimizers.Adam(lr=0.0001))
model.fit(training_data, training_labels, epochs=500, validation_data=(test_data, test_labels), batch_size=128)

plot_history([('model', model.history)])
plot_history([('model', model.history)], key='loss')
