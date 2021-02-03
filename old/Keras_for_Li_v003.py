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
        threshold = 0.95
        if (logs.get('accuracy')>threshold and logs.get('val_accuracy') > threshold):
            print("TARGET REACHED")
            self.model.stop_training = True
            self.model.save(f"models\\abx\\v003.{epoch}.accu{logs.get('accuracy'):.3f}valid{logs.get('val_accuracy'):.3f}")


callbacks = [reachedAccurCallback()]

training_data = np.load("data\\training_data_v003.2.npy")
training_labels = np.load("data\\training_labels_v003.2.npy")
test_data = np.load("data\\test_data_v003.2.npy")
test_labels = np.load("data\\test_labels_v003.2.npy")

m1 = keras.Sequential()
m1.add(keras.layers.Flatten(input_shape=(3, 3)))
m1.add(keras.layers.Dense(1500, activation='relu'))
m1.add(keras.layers.Dropout(0.40))
m1.add(keras.layers.Dense(1250, activation='relu'))
m1.add(keras.layers.Dropout(0.40))
m1.add(keras.layers.Dense(1000, activation='relu'))
m1.add(keras.layers.Dropout(0.40))
m1.add(keras.layers.Dense(750, activation='relu'))
m1.add(keras.layers.Dropout(0.40))
m1.add(keras.layers.Dense(500, activation='relu'))
m1.add(keras.layers.Dropout(0.40))
m1.add(keras.layers.Dense(1, activation='sigmoid'))

m1.compile(optimizer='Nadam',
           loss='binary_crossentropy',
           metrics=['accuracy'])
#
# m2 = keras.Sequential()
# m2.add(keras.layers.Flatten(input_shape=(3, 3)))
# m2.add(keras.layers.Dense(1200, activation='relu'))
# m2.add(keras.layers.Dropout(0.50))
# m2.add(keras.layers.Dense(1000, activation='relu'))
# m2.add(keras.layers.Dropout(0.50))
# m2.add(keras.layers.Dense(800, activation='relu'))
# m2.add(keras.layers.Dropout(0.50))
# m2.add(keras.layers.Dense(600, activation='relu'))
# m2.add(keras.layers.Dropout(0.50))
# m2.add(keras.layers.Dense(1, activation='sigmoid'))
#
# m2.compile(optimizer='Adam',
#            loss='binary_crossentropy',
#            metrics=['accuracy'])

# looks like an optimal size for 3x3 input
# m3 = keras.Sequential()
# m3.add(keras.layers.Flatten(input_shape=(3, 3)))
# m3.add(keras.layers.Dense(600, activation='relu'))
# m3.add(keras.layers.Dropout(0.40))
# m3.add(keras.layers.Dense(500, activation='relu'))
# m3.add(keras.layers.Dropout(0.40))
# m3.add(keras.layers.Dense(400, activation='relu'))
# m3.add(keras.layers.Dropout(0.40))
# m3.add(keras.layers.Dense(300, activation='relu'))
# m3.add(keras.layers.Dropout(0.40))
# m3.add(keras.layers.Dense(1, activation='sigmoid'))
#
# m3.compile(optimizer='Adam',
#            loss='binary_crossentropy',
#            metrics=['accuracy'])

#m1.summary()
m1.fit(training_data, training_labels, validation_data=(test_data, test_labels), epochs=1500, callbacks=callbacks)
# m2.fit(training_data, training_labels, validation_data=(test_data, test_labels), epochs=500)
# m3.fit(training_data, training_labels, validation_data=(test_data, test_labels), epochs=2000, callbacks=callbacks)
# plot_history([('m1', m1.history),
#               ('m2', m2.history),
#               ('m3', m3.history)])
# plot_history([('m1', m1.history),
#               ('m2', m2.history),
#               ('m3', m3.history)], key='loss')

plot_history([('m1', m1.history)])
plot_history([('m1', m1.history)], key='loss')


#eval=model.evaluate(test_data, test_labels)
#model.save(f"models\\v003.accu{eval[1]:.2f}")