import tensorflow as tf
from tensorflow import keras
import numpy as np
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
import time

start_time = time.time()

training_data = np.load("data\\training_data_v003.npy")
training_labels = np.load("data\\training_labels_v003.npy")
test_data = np.load("data\\test_data_v003.npy")
test_labels = np.load("data\\test_labels_v003.npy")

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(3, 3)))
    for n in range(0, hp.Int(name='n_layers', min_value=1, max_value=5)):
        model.add(keras.layers.Dense(hp.Int(name=f'{n}_layer_neurons', min_value=5, max_value=250, step=5), activation='relu'))
        model.add(keras.layers.Dropout(0.30))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

tuner = BayesianOptimization(
    build_model,
    objective='accuracy',
    max_trials=1000,
    directory='v003_auto_dir',
)
tuner.search(training_data, training_labels, epochs=25)
tuner.results_summary()
models = tuner.get_best_models(num_models=3)
for model in models:
    model.summary()
    model.evaluate(test_data, test_labels)
    model.evaluate(training_data, training_labels)
print("--- %s seconds ---" % (time.time() - start_time))