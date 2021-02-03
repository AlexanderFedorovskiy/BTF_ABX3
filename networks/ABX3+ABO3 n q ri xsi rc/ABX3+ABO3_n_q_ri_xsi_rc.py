import numpy as np
import networks.seqnet as sq
import DataTransformer as dt
from kerastuner.tuners import BayesianOptimization


# load data #
path = "C:\\Users\\rustahk\\Documents\\My Projects\\PerovClass\\"
# training_data_abo = np.load(path+"datasets\\training_data.ABO Li v001.v001.npy")
# training_labels_abo = np.load(path+"datasets\\training_labels.ABO Li v001.v001.npy")
# test_data_abo = np.load(path+"datasets\\test_data.ABO Li v001.v001.npy")
# test_labels_abo = np.load(path+"datasets\\test_labels.ABO Li v001.v001.npy")
# training_data_abx = np.load(path+"datasets\\training_data.ABX Li v001.v001.npy")
# training_labels_abx = np.load(path+"datasets\\training_labels.ABX Li v001.v001.npy")
# test_data_abx = np.load(path + "datasets\\test_data.ABX Li v001.v001.npy")
# test_labels_abx = np.load(path+"datasets\\test_labels.ABX Li v001.v001.npy")
####
training_data_abo = np.load(path+"datasets\\training_data.ABO Li v002.v001.npy")
training_labels_abo = np.load(path+"datasets\\training_labels.ABO Li v002.v001.npy")
test_data_abo = np.load(path+"datasets\\test_data.ABO Li v002.v001.npy")
test_labels_abo = np.load(path+"datasets\\test_labels.ABO Li v002.v001.npy")
training_data_abx = np.load(path+"datasets\\training_data.ABX Li v002.v001.npy")
training_labels_abx = np.load(path+"datasets\\training_labels.ABX Li v002.v001.npy")
test_data_abx = np.load(path + "datasets\\test_data.ABX Li v002.v001.npy")
test_labels_abx = np.load(path+"datasets\\test_labels.ABX Li v002.v001.npy")
####
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


tuner = BayesianOptimization(
    sq.build_model_hp_3x5,
    objective='val_accuracy',
    max_trials=250,
    num_initial_points=10,
    directory='D:\\models\\ABX3+ABO3 n q ri xsi rc Bayes',
    project_name='pyramid_epoch1000'
)

tuner.search(training_data, training_labels, epochs=750, validation_data=(test_data, test_labels), batch_size=64)
tuner.results_summary()
models = tuner.get_best_models(num_models=10)
for iteration, model in enumerate(models):
    print('n:', iteration)
    model.summary()
    print("Test: ")
    model.evaluate(test_data, test_labels)
    print("Training: ")
    model.evaluate(training_data, training_labels)
    if iteration == 6:
        model.save("M_abx3+abo3_n_q_ri_xsi_rc_tr96.250ts95.65")
    # if iteration == 34:
    #     model.save("M_abo3_n_q_ri_xsi_rc_tr97.08ts98.31")
