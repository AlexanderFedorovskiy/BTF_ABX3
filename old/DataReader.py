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
        tempcompnents.append(int(templist[3]))
        rawdata.append(tempcompnents)
    rawdata = np.asarray(rawdata, dtype=np.int16)
    np.random.shuffle(rawdata)
    a, b = rawdata.shape
    border = int(a*0.7) #for v <.2 it was 0.8
    training, test = rawdata[:border, :], rawdata[border:, :]
    training_labels = training[:, -1]
    training_data = training[:, :3]
    test_labels = test[:, -1]
    test_data = test[:, :3]
    return training_data, training_labels, test_data, test_labels


def savetofile(data, filename):
    np.save(f'{filename}.npy', data)


def makedata():
    training_data, training_labels, test_data, test_labels = readrawfile("data\\dataset ABX Li v001.txt")
    savetofile(training_data, 'data\\training_data_v001.2')
    savetofile(training_labels, 'data\\training_labels_v001.2')
    savetofile(test_data, 'data\\test_data_v001.2')
    savetofile(test_labels, 'data\\test_labels_v001.2')

makedata()
