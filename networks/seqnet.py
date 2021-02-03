import numpy as np
from tensorflow import keras

def build_model(n_layers, n_neurons, flattern=False, shape=None, top_value = 0.1 ,input_dim=3, input_shape=(3, 3), activation='relu', kernel_initializer='uniform', optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy']):
    # top_value is a percent of size for last hidden layer (first layer is 100% = 1)
    model = keras.Sequential()
    if not n_layers == 1:
        step = (1 - top_value) / (n_layers-1)
    for n in range(0, n_layers):
        if n == 0:
            if flattern:
                model.add(keras.layers.Flatten(input_shape=shape))
                model.add(keras.layers.Dense(units=n_neurons, activation=activation, kernel_initializer=kernel_initializer))
            else:
                model.add(keras.layers.Dense(units=n_neurons, activation=activation,kernel_initializer=kernel_initializer, input_dim=input_dim))
        else:
            percentage = 1-step*n
            number = int(n_neurons*percentage)
            if number <= 1:
                number = 2
            model.add(keras.layers.Dense(units=number, activation=activation, kernel_initializer=kernel_initializer))
        model.add(keras.layers.Dropout(0.50))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer, loss, metrics)
    return model


def build_model_hp(hp):
    n_layers = hp.Int(name='n_layers', min_value=1, max_value=10, step=1)
    n_neurons = hp.Int(name=f'n_neurons', min_value=10, max_value=1000, step=10)
    model = build_model(n_layers, n_neurons)
    return model


def build_model_hp_3x2(hp):
    n_layers = hp.Int(name='n_layers', min_value=1, max_value=10, step=1)
    n_neurons = hp.Int(name=f'n_neurons', min_value=10, max_value=1000, step=10)
    model = build_model(n_layers, n_neurons, flattern=True, shape=(3, 2))
    return model


def build_model_hp_3x3(hp):
    n_layers = hp.Int(name='n_layers', min_value=1, max_value=10, step=1)
    n_neurons = hp.Int(name=f'n_neurons', min_value=10, max_value=1000, step=10)
    model = build_model(n_layers, n_neurons, flattern=True, shape=(3, 3))
    return model


def build_model_hp_3x4(hp):
    n_layers = hp.Int(name='n_layers', min_value=1, max_value=10, step=1)
    n_neurons = hp.Int(name=f'n_neurons', min_value=10, max_value=1000, step=10)
    model = build_model(n_layers, n_neurons, flattern=True, shape=(3, 4))
    return model


def build_model_hp_3x5(hp):
    n_layers = hp.Int(name='n_layers', min_value=1, max_value=10, step=1)
    n_neurons = hp.Int(name=f'n_neurons', min_value=10, max_value=1000, step=10)
    model = build_model(n_layers, n_neurons, flattern=True, shape=(3, 5))
    return model



def writemetcics(parameters, accuracies, losses, val_accuracies, val_losses):
    file = open(f"{parameters}.txt", "x")
    for i in range(0, len(accuracies)):
        file.write(f"{i}\t{accuracies[i]}\t{losses[i]}\t{val_accuracies[i]}\t{val_losses[i]}\n")
    file.close()