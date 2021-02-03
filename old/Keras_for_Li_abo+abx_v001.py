import tensorflow as tf
from tensorflow import keras
import numpy as np
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
        threshold = 0.92
        if (logs.get('accuracy')>threshold and logs.get('val_accuracy') > threshold):
            print("TARGET REACHED")
            self.model.stop_training = True
            self.model.save(f"models\\abo+abx\\{self.model.name}.v001+cov.{epoch}.accu{logs.get('accuracy'):.3f}valid{logs.get('val_accuracy'):.3f}")


callbacks = [reachedAccurCallback()]

training_data = np.load("data\\ABO+ABX_training_data_v002.npy")
training_labels = np.load("data\\ABO+ABX_training_labels_v002.npy")
test_data = np.load("data\\ABO+ABX_test_data_v002.npy")
test_labels = np.load("data\\ABO+ABX_test_labels_v002.npy")

training_data_cov = np.load("data\\ABO+ABX_training_data_v002+cov.npy")
training_labels_cov = np.load("data\\ABO+ABX_training_labels_v002+cov.npy")
test_data_cov = np.load("data\\ABO+ABX_test_data_v002+cov.npy")
test_labels_cov = np.load("data\\ABO+ABX_test_labels_v002+cov.npy")

# m1 = keras.Sequential()
# m1.add(keras.layers.Flatten(input_shape=(3, 4)))
# m1.add(keras.layers.Dense(150, activation='relu'))
# m1.add(keras.layers.Dropout(0.40))
# m1.add(keras.layers.Dense(140, activation='relu'))
# m1.add(keras.layers.Dropout(0.40))
# m1.add(keras.layers.Dense(130, activation='relu'))
# m1.add(keras.layers.Dropout(0.40))
# m1.add(keras.layers.Dense(120, activation='relu'))
# m1.add(keras.layers.Dropout(0.40))
# m1.add(keras.layers.Dense(110, activation='relu'))
# m1.add(keras.layers.Dropout(0.40))
# m1.add(keras.layers.Dense(100, activation='relu'))
# m1.add(keras.layers.Dropout(0.40))
# m1.add(keras.layers.Dense(90, activation='relu'))
# m1.add(keras.layers.Dropout(0.40))
# m1.add(keras.layers.Dense(80, activation='relu'))
# m1.add(keras.layers.Dropout(0.40))
# m1.add(keras.layers.Dense(70, activation='relu'))
# m1.add(keras.layers.Dropout(0.40))
# m1.add(keras.layers.Dense(60, activation='relu'))
# m1.add(keras.layers.Dropout(0.40))
# m1.add(keras.layers.Dense(1, activation='sigmoid'))
#
# m1.compile(optimizer='Nadam',
#            loss='binary_crossentropy',
#            metrics=['accuracy'])

m2 = keras.Sequential()
m2.add(keras.layers.Flatten(input_shape=(3, 5)))
m2.add(keras.layers.Dense(200, activation='relu'))
m2.add(keras.layers.Dropout(0.40))
m2.add(keras.layers.Dense(190, activation='relu'))
m2.add(keras.layers.Dropout(0.40))
m2.add(keras.layers.Dense(180, activation='relu'))
m2.add(keras.layers.Dropout(0.40))
m2.add(keras.layers.Dense(170, activation='relu'))
m2.add(keras.layers.Dropout(0.40))
m2.add(keras.layers.Dense(160, activation='relu'))
m2.add(keras.layers.Dropout(0.40))
m2.add(keras.layers.Dense(150, activation='relu'))
m2.add(keras.layers.Dropout(0.40))
m2.add(keras.layers.Dense(140, activation='relu'))
m2.add(keras.layers.Dropout(0.40))
m2.add(keras.layers.Dense(130, activation='relu'))
m2.add(keras.layers.Dropout(0.40))
m2.add(keras.layers.Dense(120, activation='relu'))
m2.add(keras.layers.Dropout(0.40))
m2.add(keras.layers.Dense(110, activation='relu'))
m2.add(keras.layers.Dropout(0.40))
m2.add(keras.layers.Dense(1, activation='sigmoid'))

m2.compile(optimizer='Adam',
           loss='binary_crossentropy',
           metrics=['accuracy'])

# m3 = keras.Sequential()
# m3.add(keras.layers.Flatten(input_shape=(3, 4)))
# m3.add(keras.layers.Dense(1500, activation='relu'))
# m3.add(keras.layers.Dropout(0.40))
# m3.add(keras.layers.Dense(1000, activation='relu'))
# m3.add(keras.layers.Dropout(0.40))
# m3.add(keras.layers.Dense(750, activation='relu'))
# m3.add(keras.layers.Dropout(0.40))
# m3.add(keras.layers.Dense(500, activation='relu'))
# m3.add(keras.layers.Dropout(0.40))
# m3.add(keras.layers.Dense(250, activation='relu'))
# m3.add(keras.layers.Dropout(0.40))
# m3.add(keras.layers.Dense(1, activation='sigmoid'))

# m3.compile(optimizer='Adam',
#            loss='binary_crossentropy',
#            metrics=['accuracy'])

# m1.fit(training_data, training_labels, validation_data=(test_data, test_labels), epochs=1200, callbacks=callbacks)
m2.fit(training_data_cov, training_labels_cov, validation_data=(test_data_cov, test_labels_cov), epochs=2000, callbacks=callbacks)
# m3.fit(training_data, training_labels, validation_data=(test_data, test_labels), epochs=400)
# plot_history([('m1', m1.history),
#               ('m2', m2.history)])
# plot_history([('m1', m1.history),
#               ('m2', m2.history)], key='loss')
#
plot_history([('m2', m2.history)])
# plot_history([('m1', m1.history)], key='loss')

#eval=model.evaluate(test_data, test_labels)
#model.save(f"models\\v003.accu{eval[1]:.2f}")