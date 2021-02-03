import numpy as np
import mendeleev as md
from old import DataReader as dr


# Change ndarray of input ABX3 element numbers to its radii
def cast_to_radii(data, labels):
    merged = np.insert(data, 3, labels, axis=1)
    for row in merged:
        radii_in_row(row)
    merged = merged[np.all(merged >= 0, axis=1), :]  # remove line if radii(A) <= 0
    data = merged[:, :3]
    labels = merged[:, -1]
    return data, labels


def radii_in_row(line):
    for i in range(line.size-1):
        temp = atoms_to_radii(line[i], i)
        line[i] = temp
    return line


def atoms_to_radii(elementnumber, position):
    unit = md.element(int(elementnumber))
    cordn = coordination(position)
    for ir in unit.ionic_radii:
        if ir.coordination==cordn:
            return ir.ionic_radius
    print(f"WARNING, NO VALUE FOR: {elementnumber}, {position}. THE COMPOUND WILL BE IGNORED")
    return -1

def coordination(position):
    argument = {
        0: "XII",
        1: "VI",
        2: "VI"
    }
    return argument.get(position)


def makedata():
    test_data, test_labels = cast_to_radii(np.load("data\\test_data_v001.2.npy"), np.load("data\\test_labels_v001.2.npy"))
    training_data, training_labels = cast_to_radii(np.load("data\\training_data_v001.2.npy"), np.load("data\\training_labels_v001.2.npy"))
    dr.savetofile(training_data, 'data\\training_data_v002.2')
    dr.savetofile(training_labels, 'data\\training_labels_v002.2')
    dr.savetofile(test_data, 'data\\test_data_v002.2')
    dr.savetofile(test_labels, 'data\\test_labels_v002.2')

makedata()