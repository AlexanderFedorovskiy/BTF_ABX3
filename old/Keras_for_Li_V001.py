import tensorflow as tf
from tensorflow import keras
import numpy as np

# training_data = np.load("data\\training_data_v001.2.npy")
# training_labels = np.load("data\\training_labels_v001.2.npy")
# test_data = np.load("data\\test_data_v001.2.npy")
# test_labels = np.load("data\\test_labels_v001.2.npy")

training_data = np.load("data\\abo_training_data_v001.npy")
training_labels = np.load("data\\abo_training_labels_v001.npy")
test_data = np.load("data\\abo_test_data_v001.npy")
test_labels = np.load("data\\abo_test_labels_v001.npy")
training_data = np.delete(training_data, 3, 1)
test_data = np.delete(test_data, 3, 1)

training_data = training_data/100
test_data = test_data/100

model = keras.Sequential()
model.add(tf.keras.layers.Dense(30, activation='relu', input_dim=3, kernel_initializer='uniform'))
model.add(tf.keras.layers.Dropout(0.50))
model.add(tf.keras.layers.Dense(600, activation='relu', kernel_initializer='uniform'))
model.add(tf.keras.layers.Dropout(0.50))
model.add(tf.keras.layers.Dense(600, activation='relu', kernel_initializer='uniform'))
model.add(tf.keras.layers.Dropout(0.50))
model.add(tf.keras.layers.Dense(600, activation='relu', kernel_initializer='uniform'))
model.add(tf.keras.layers.Dropout(0.50))
model.add(tf.keras.layers.Dense(1, kernel_initializer='uniform',activation='sigmoid'))

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(training_data, training_labels, epochs=500, validation_data=(test_data, test_labels))
# model.evaluate(test_data, test_labels)
#model.summary()