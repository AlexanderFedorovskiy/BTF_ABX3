from chempy import Substance
import numpy as np


def readrawfile(filename):
    file = open(filename)
    rawdata = []
    for line in file:
        templine = line.rstrip('\n')
        templist = templine.split('\t')
        formula = Substance.from_formula(templist[0] + templist[1] + templist[2] + '3')
        tempcompnents = []
        for i in formula.composition:
            tempcompnents.append(int(i))
        tempcompnents.append(perovskite_group(templist[4]))  # Be aware, inverted columns!
        tempcompnents.append(int(templist[3]))
        rawdata.append(tempcompnents)
    rawdata = np.asarray(rawdata, dtype=np.int16)
    np.random.shuffle(rawdata)
    a, b = rawdata.shape
    border = int(a * 0.7)
    training, test = rawdata[:border, :], rawdata[border:, :]
    training_labels = training[:, -1]
    training_data = training[:, :4]
    test_labels = test[:, -1]
    test_data = test[:, :4]
    return training_data, training_labels, test_data, test_labels


def savetofile(data, filename):
    np.save(f'{filename}.npy', data)


def perovskite_group(label):
    if label == "A2O-B2O5":
        return 0
    elif label == "AO-BO2":
        return 1
    elif label == "A2O3-B2O3":
        return 2
    elif label == "ABX3":
        return 3
    return -1


def makedata():
    training_data, training_labels, test_data, test_labels = readrawfile("data\\dataset ABO+ABX Li v001.txt")
    savetofile(training_data, 'data\\ABO+ABX_training_data_v001')
    savetofile(training_labels, 'data\\ABO+ABX_training_labels_v001')
    savetofile(test_data, 'data\\ABO+ABX_test_data_v001')
    savetofile(test_labels, 'data\\ABO+ABX_test_labels_v001')


makedata()
