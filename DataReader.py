from chempy import Substance
import numpy as np


# Input format: ID A_element B_element X_element Group Label
def readrawfile(filename):
    file = open(filename)
    data = []
    for line in file:
        if(line[0] == '#'):
            continue
        templine = line.rstrip('\n')
        templist = templine.split('\t')
        formula = Substance.from_formula(templist[1] + templist[2] + templist[3])
        tempcompnents = []
        tempcompnents.append(castidtoint(templist[0]))
        for i in formula.composition:
            tempcompnents.append(int(i))
        tempcompnents.append(perovskite_group(templist[4]))
        tempcompnents.append(int(templist[5]))
        data.append(tempcompnents)
    data = np.asarray(data, dtype=np.int16)
    return data


def castidtoint(id):
    id = id.replace('ID', '')
    temp = id.split('.')
    return int(temp[0]+'0'+temp[1])


def savetofile(data, name):
    np.save(f'datasets//{name}.npy', data)


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


def split(data):
    np.random.shuffle(data)
    a, b = data.shape
    border = int(a * 0.7)  # 0.7 is the split ratio
    training, test = data[:border, :], data[border:, :]
    training_labels = training[:, -1]
    training_data = training[:, :5]
    test_labels = test[:, -1]
    test_data = test[:, :5]
    return training_data, training_labels, test_data, test_labels


def savesplitteddata(training_data, training_labels, test_data, test_labels, dataname, splitnumber):
    savetofile(training_data, "training_data."+dataname+".v"+splitnumber)
    savetofile(training_labels, "training_labels."+dataname+".v"+splitnumber)
    savetofile(test_data, "test_data."+dataname+".v"+splitnumber)
    savetofile(test_labels, "test_labels."+dataname+".v"+splitnumber)
