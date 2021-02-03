import numpy as np
import mendeleev as md


def cast_to_properties(data):
    novel_data = []
    for row in data:
        line = []
        for i in range(row.size - 1):
            prop_vector = []
            atom_number = row[i]
            charge = get_charge(i, row[3])
            ionic_radii = get_ionic_radii(row[i], i, charge)
            elneg = get_electronegativity(row[i])
            cov_radii = get_cov_radii(row[i])
            prop_vector.append(atom_number)  # n 0
            prop_vector.append(charge)  # q 1
            prop_vector.append(ionic_radii)  # rii 2
            prop_vector.append(elneg)  # xsi 3
            prop_vector.append(cov_radii)  # rc 4
            line.append(prop_vector)
        novel_data.append(line)
    novel_data = np.asarray(novel_data, dtype=float)
    return novel_data


def minmax(value, min, max):
    return (value - min) / (max - min)


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


def get_cov_radii(elementnumber):
    unit = md.element(int(elementnumber))
    return unit.covalent_radius_cordero


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
    arg_g3 = {
        0: 1,
        1: 2,
        2: -1
    }
    if group == 0:
        return arg_g0.get(position)
    elif group == 1:
        return arg_g1.get(position)
    elif group == 2:
        return arg_g2.get(position)
    elif group == 3:
        return arg_g3.get(position)
    else:
        return None


def get_coordination(position):
    argument = {
        0: "XII",
        1: "VI",
        2: "VI"
    }
    return argument.get(position)


def get_ionic_radii(elementnumber, position, charge):
    unit = md.element(int(elementnumber))
    coordination = get_coordination(position)
    # based on Li's ABO3 paper
    if elementnumber == 70 and position == 0 and charge == 3:  # Yb
        return 86.0
    elif elementnumber == 64 and position == 0 and charge == 3:  # Gd
        return 94.0
    elif elementnumber == 3 and position == 0 and charge == 1:  # Li
        return 76.0  # Alternative value: 119
    elif elementnumber == 59 and position == 0 and charge == 3:  # Pr
        return 99.0
    elif elementnumber == 47 and position == 0 and charge == 1:  # Ag
        return 115.0
    elif elementnumber == 23 and position == 0 and charge == 3:  # V
        return 64.0
    elif elementnumber == 71 and position == 0 and charge == 3:  # Lu
        return 86.0
    elif elementnumber == 63 and position == 0 and charge == 3:  # Eu
        return 95.0
    elif elementnumber == 83 and position == 0 and charge == 3:  # Bi
        return 103.0
    elif elementnumber == 65 and position == 0 and charge == 3:  # Tb
        return 92.0
    elif elementnumber == 13 and position == 0 and charge == 3:  # Al
        return 54.0
    elif elementnumber == 33 and position == 0 and charge == 3:  # As
        return 58.0
    elif elementnumber == 39 and position == 0 and charge == 3:  # Y
        return 90.0
    elif elementnumber == 21 and position == 0 and charge == 3:  # Sc
        return 75.0
    elif elementnumber == 68 and position == 0 and charge == 3:  # Er
        return 89.0
    elif elementnumber == 50 and position == 0 and charge == 2:  # Sn
        return 93.0
    elif elementnumber == 67 and position == 0 and charge == 3:  # Ho
        return 89.0
    elif elementnumber == 12 and position == 0 and charge == 2:  # Mg
        return 72.0
    elif elementnumber == 26 and position == 0 and charge == 2:  # Fe
        return 61.0
    elif elementnumber == 30 and position == 0 and charge == 2:  # Zn
        return 74.0
    elif elementnumber == 31 and position == 0 and charge == 3:  # Ga
        return 62.0
    elif elementnumber == 66 and position == 0 and charge == 3:  # Dy
        return 91.0
    elif elementnumber == 27 and position == 0 and charge == 2:  # Co
        return 65.0
    elif elementnumber == 63 and position == 0 and charge == 2:  # Eu
        return 117.0
    elif elementnumber == 49 and position == 0 and charge == 3:  # In
        return 80.0
    elif elementnumber == 69 and position == 0 and charge == 3:  # Tm
        return 88.0
    elif elementnumber == 28 and position == 0 and charge == 2:  # Ni
        return 69.0
    elif elementnumber == 25 and position == 0 and charge == 2:  # Mn
        return 83.0
    elif elementnumber == 62 and position == 0 and charge == 2:  # Sm
        return 119.0
    # based on Li's ABX3 paper
    elif elementnumber == 53 and position == 2 and charge == -2:  # I
        return 220.0
    elif elementnumber == 29 and position == 0 and charge == 1:  # Cu
        return 110.0
    elif elementnumber == 40 and position == 1 and charge == 2:  # Zr
        return 72.0
    elif elementnumber == 50 and position == 1 and charge == 2:  # Sn
        return 110.0
    elif elementnumber == 62 and position == 1 and charge == 2:  # Sm
        return 122.0
    for ir in unit.ionic_radii:
        if ir.coordination == coordination and ir.charge == charge:
            return ir.ionic_radius
    print(f"ERROR, NO RADII VALUE FOR: {elementnumber}, {position}, {charge}")
    return -1.0


def normalize(data):
    data[:, :, 0] = minmax(data[:, :,  0], 0.0, 100.0)  # up to Fermium n=100
    data[:, :, 1] = minmax(data[:, :, 1], -2.0, 5.0)  # up to A2O-B2O5 where B+5
    data[:, :, 2] = minmax(data[:, :, 2], 1.0, 221.0)  # Up to Te_raii=221
    data[:, :, 3] = minmax(data[:, :, 3], 0.79, 3.98)  # Between Cs = 0.79 & F = 3.98
    data[:, :, 4] = minmax(data[:, :, 4], 1.0, 260.0)  # up to Fr
    return data
