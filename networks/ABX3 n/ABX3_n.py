import numpy as np
import networks.seqnet as sq
from tensorflow import keras
from kerastuner.tuners import BayesianOptimization

def writemetcics(parameters, accuracies, losses, val_accuracies, val_losses):
    file = open(f"{parameters}.txt", "x")
    for i in range(0, len(accuracies)):
        file.write(f"{i}\t{accuracies[i]}\t{losses[i]}\t{val_accuracies[i]}\t{val_losses[i]}\n")
    file.close()


def build_model_hp(hp):
    n_layers = hp.Int(name='n_layers', min_value=1, max_value=10)
    n_neurons = hp.Int(name=f'n_neurons', min_value=10, max_value=1000, step=10)
    model = sq.build_model(n_layers, n_neurons)
    return model


def build_model(n_layers, n_neurons):
    model = sq.build_model(n_layers, n_neurons)
    return model

# load data # todo remake file upload
path = "C:\\Users\\rustahk\\Documents\\My Projects\\PerovClass\\"
training_data = np.load(path+"datasets\\training_data.ABX Li v001.v001.npy")
training_labels = np.load(path+"datasets\\training_labels.ABX Li v001.v001.npy")
test_data = np.load(path+"datasets\\test_data.ABX Li v001.v001.npy")
test_labels = np.load(path+"datasets\\test_labels.ABX Li v001.v001.npy")
# remove id, group
training_data = np.delete(training_data, [0, 4], 1)
test_data = np.delete(test_data, [0, 4], 1)
# normalization
training_data = training_data/100
test_data = test_data/100

tuner = BayesianOptimization(
    build_model_hp,
    objective='val_accuracy',
    max_trials=1000,
    num_initial_points=10,
    directory='D:\\models\\ABX3 n Bayes',
    project_name='pyramid'
)

tuner.search(training_data, training_labels, epochs=500, validation_data=(test_data, test_labels), batch_size=64)
tuner.results_summary()
models = tuner.get_best_models(num_models=10)
for iteration, model in enumerate(models):
    print('n:', iteration)
    model.summary()
    print("Test: ")
    model.evaluate(test_data, test_labels)
    print("Training: ")
    model.evaluate(training_data, training_labels)
    # if iteration == 1:
    #     model.save("M_abx3_n_tr82.31ts87.5")