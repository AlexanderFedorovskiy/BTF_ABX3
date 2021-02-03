import numpy as np
import networks.seqnet as sq
from tensorflow import keras
import DataTransformer as dt
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

class reachedAccurCallback(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
          threshold = 0.98
          if (logs.get('accuracy') > threshold and logs.get('val_accuracy') > threshold):
              print("TARGET REACHED")
              self.model.stop_training = True
              self.model.save(
                  f"models\\abx\\v003.{epoch}.accu{logs.get('accuracy'):.3f}valid{logs.get('val_accuracy'):.3f}")


callbacks = [reachedAccurCallback()]

# load data #
path = "C:\\Users\\rustahk\\Documents\\My Projects\\PerovClass\\"
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

optimizer_small = keras.optimizers.Adam(lr=0.0001)

model = sq.build_model(n_neurons=1500, n_layers=5, flattern=True, shape=(3,5), optimizer=optimizer_small)
model.fit(training_data, training_labels, validation_data=(test_data, test_labels), epochs=2000, callbacks=callbacks, batch_size=128)

plot_history([('m', model.history)])
plot_history([('m', model.history)], key='loss')
