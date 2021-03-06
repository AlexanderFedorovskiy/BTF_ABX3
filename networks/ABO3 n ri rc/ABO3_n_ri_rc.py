import numpy as np
import networks.seqnet as sq
import DataTransformer as dt
from kerastuner.tuners import BayesianOptimization


# load data #
path = "C:\\Users\\rustahk\\Documents\\My Projects\\PerovClass\\"
training_data = np.load(path+"datasets\\training_data.ABO Li v001.v001.npy")
training_labels = np.load(path+"datasets\\training_labels.ABO Li v001.v001.npy")
test_data = np.load(path+"datasets\\test_data.ABO Li v001.v001.npy")
test_labels = np.load(path+"datasets\\test_labels.ABO Li v001.v001.npy")
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
training_data = np.delete(training_data, [1, 3], 2)
test_data = np.delete(test_data, [1, 3], 2)


tuner = BayesianOptimization(
    sq.build_model_hp_3x3,
    objective='val_accuracy',
    max_trials=500,
    num_initial_points=10,
    directory='D:\\models\\ABO3 n ri rc Bayes',
    project_name='pyramid'
)

tuner.search(training_data, training_labels, epochs=500, validation_data=(test_data, test_labels), batch_size=64)
tuner.results_summary()
models = tuner.get_best_models(num_models=30)
for iteration, model in enumerate(models):
    print('n:', iteration)
    model.summary()
    print("Test: ")
    model.evaluate(test_data, test_labels)
    print("Training: ")
    model.evaluate(training_data, training_labels)
    # if iteration == 9:
    #      model.save("M_abo3_n_ri_rc_tr94.89ts100")
    # if iteration == 20:
    #     model.save("M_abo3_n_ri_rc_tr97.81ts100")
    # if iteration == 26:
    #     model.save("M_abo3_n_ri_rc_tr97.81ts98.31")

