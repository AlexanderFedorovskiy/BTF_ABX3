import numpy as np
import mendeleev as md


# Change ndarray of input ABO3 element numbers to its properties
def cast_to_properties(data, labels):
    novel_data = []
    for row in data:
        line = []
        for i in range(row.size - 1):
            prop_vector = []
            atom_number = normalize(row[i], 0, 100.0)  # up to Fermium n=100
            charge = get_charge(i, row[3])
            radii = normalize(get_radii(row[i], i, charge), 1.0, 221.0)  # Up to Te_raii=221
            elneg = normalize(get_electronegativity(row[i]), 0.79, 3.98)  # Normalization (Between Cs = 0.79 & F = 3.98)
            charge = normalize(charge, -2.0, 5.0)  # up to A2O-B2O5 where B+5
            prop_vector.append(atom_number)
            prop_vector.append(charge)
            prop_vector.append(radii)
            prop_vector.append(elneg)
            line.append(prop_vector)
        novel_data.append(line)
    novel_data = np.asarray(novel_data, dtype=float)
    return novel_data, labels


def normalize(value, min, max):
    return (value-min)/(max-min)

def get_electronegativity(elementnumber):
    unit = md.element(int(elementnumber))
    elneg = unit.electronegativity(scale='pauling')
    if elneg == None:
        if elementnumber == 63:  # Eu
            elneg = 1.2
        elif elementnumber == 65:  # Tb
            elneg = 1.2
        elif elementnumber == 70:  # Yb
            elneg = 1.1
        else:
            print(f"ERROR, NO ELECTRONEGATIVITY VALUE FOR: {elementnumber}")
            elneg = 0
    return elneg


def get_radii(elementnumber, position, charge):
    unit = md.element(int(elementnumber))
    coordination = get_coordination(position)
    # based on Li's ABO3 paper
    if elementnumber==70 and position==0 and charge==3:  # Yb
        return 86.0
    elif elementnumber==64 and position==0 and charge==3:  # Gd
        return 94.0
    elif elementnumber==3 and position==0 and charge==1:  # Li
        return 76.0
    elif elementnumber==59 and position==0 and charge==3:  # Pr
        return 99.0
    elif elementnumber==47 and position==0 and charge==1:  # Ag
        return 115.0
    elif elementnumber==23 and position==0 and charge==3:  # V
        return 64.0
    elif elementnumber==71 and position==0 and charge==3:  # Lu
        return 86.0
    elif elementnumber==63 and position==0 and charge==3:  # Eu
        return 95.0
    elif elementnumber==83 and position==0 and charge==3:  # Bi
        return 103.0
    elif elementnumber==65 and position==0 and charge==3:  # Tb
        return 92.0
    elif elementnumber==13 and position==0 and charge==3:  # Al
        return 54.0
    elif elementnumber==33 and position==0 and charge==3:  # As
        return 58.0
    elif elementnumber==39 and position==0 and charge==3:  # Y
        return 90.0
    elif elementnumber==21 and position==0 and charge==3:  # Sc
        return 75.0
    elif elementnumber==68 and position==0 and charge==3:  # Er
        return 89.0
    elif elementnumber==50 and position==0 and charge==2:  # Sn
        return 93.0
    elif elementnumber==67 and position==0 and charge==3:  # Ho
        return 89.0
    elif elementnumber==12 and position==0 and charge==2:  # Mg
        return 72.0
    elif elementnumber==26 and position==0 and charge==2:  # Fe
        return 61.0
    elif elementnumber==30 and position==0 and charge==2:  # Zn
        return 74.0
    elif elementnumber==31 and position==0 and charge==3:  # Ga
        return 62.0
    elif elementnumber==66 and position==0 and charge==3:  # Dy
        return 91.0
    elif elementnumber==27 and position==0 and charge==2:  # Co
        return 65.0
    elif elementnumber==63 and position==0 and charge==2:  # Eu
        return 117.0
    elif elementnumber==49 and position==0 and charge==3:  # In
        return 80.0
    elif elementnumber==69 and position==0 and charge==3:  # Tm
        return 88.0
    elif elementnumber==28 and position==0 and charge==2:  # Ni
        return 69.0
    elif elementnumber==25 and position==0 and charge==2:  # Mn
        return 83.0
    elif elementnumber==62 and position==0 and charge==2:  # Sm
        return 119.0
    elif elementnumber==29 and position==0 and charge==1:  # Cu
        return 77.0

    #
    for ir in unit.ionic_radii:
        if ir.coordination == coordination and ir.charge == charge:
            return ir.ionic_radius
    print(f"ERROR, NO RADII VALUE FOR: {elementnumber}, {position}, {charge}")
    return -1.0


def get_coordination(position):
    argument = {
        0: "XII",
        1: "VI",
        2: "VI"
    }
    return argument.get(position)


def get_charge(position, group):
    arg_g0 = {
        0: 1,
        1: 5,
        2: -2
    }
    arg_g1 = {
        0: 2,
        1: 4,
        2: -2
    }
    arg_g2 = {
        0: 3,
        1: 3,
        2: -2
    }
    if group == 0:
        return arg_g0.get(position)
    elif group == 1:
        return arg_g1.get(position)
    elif group == 2:
        return arg_g2.get(position)
    else:
        return None


def makedata():
    test_data, test_labels = cast_to_properties(np.load("data\\abo_test_data_v001.npy"),
                                                np.load("data\\abo_test_labels_v001.npy"))
    training_data, training_labels = cast_to_properties(np.load("data\\abo_training_data_v001.npy"),
                                                        np.load("data\\abo_training_labels_v001.npy"))
    np.save('data\\abo_training_data_v002.npy', training_data)
    np.save('data\\abo_training_labels_v002.npy', training_labels)
    np.save('data\\abo_test_data_v002.npy', test_data)
    np.save('data\\abo_test_labels_v002.npy', test_labels)
    print("Done")


makedata()
