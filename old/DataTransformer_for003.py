import numpy as np
import mendeleev as md


# Change ndarray of input ABX3 element numbers to its properties
def cast_to_properties(data, labels):
    novel_data = []
    for row in data:
        line = []
        for i in range(row.size):
            prop_vector = []
            atom_number = row[i]/100  # Normalization (up to Fermium n=100)
            radii = atoms_to_radii(row[i], i)/220  # Normalization (up to I_raii=220)
            elneg = atoms_to_electronegativity(row[i])/3.98  # Normalization (up to F = 3.98)
            prop_vector.append(atom_number)
            prop_vector.append(radii)
            prop_vector.append(elneg)
            line.append(prop_vector)
        novel_data.append(line)
    novel_data = np.asarray(novel_data, dtype=float)
    return novel_data, labels

def atoms_to_electronegativity(elementnumber):
    unit = md.element(int(elementnumber))
    elneg = unit.electronegativity(scale='pauling')
    if elneg==None:
        if elementnumber==63:  # Eu
            elneg=1.2
        else:
            print(f"ERROR, NO RADII VALUE FOR: {elementnumber}")
            elneg=0
    return elneg

def atoms_to_radii(elementnumber, position):
    unit = md.element(int(elementnumber))
    cordn = coordination(position)
    if elementnumber==3:  # Li
        return 119.0
    elif elementnumber==29:  # Cu
        return 103.0
    elif elementnumber==47:  # Ag
        return 143.0
    elif elementnumber==53:  # I
        return 220.0
    for ir in unit.ionic_radii:
        if ir.coordination==cordn:
            return ir.ionic_radius
    print(f"ERROR, NO RADII VALUE FOR: {elementnumber}, {position}")
    return -1

def coordination(position):
    argument = {
        0: "XII",
        1: "VI",
        2: "VI"
    }
    return argument.get(position)


def makedata():
    test_data, test_labels = cast_to_properties(np.load("data\\test_data_v001.2.npy"), np.load("data\\test_labels_v001.2.npy"))
    training_data, training_labels = cast_to_properties(np.load("data\\training_data_v001.2.npy"), np.load("data\\training_labels_v001.2.npy"))
    np.save('data\\training_data_v003.2.npy', training_data)
    np.save('data\\training_labels_v003.2.npy', training_labels)
    np.save('data\\test_data_v003.2.npy', test_data)
    np.save('data\\test_labels_v003.2.npy', test_labels)
    print("Done")

makedata()