import mendeleev as md

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

def readrawfile(input, output):
    input_file = open(input)
    newfile = []
    for line in input_file:
        row = []
        templine = line.rstrip('\n')
        templist = templine.split('\t')
        for i in range(0,3):
            element = md.element(templist[i])
            row.append(element.symbol)
            row.append(atoms_to_radii(element.atomic_number, i))
        row.append(templist[3])
        newfile.append(row)
    output_file = open(output, 'x')
    for line in newfile:
        out = ""
        for cell in line:
            out+=str(cell)+';'
        out+='\n'
        output_file.write(out)
    output_file.close()
    print("here")

readrawfile("data\\dataset ABX Li v001.txt", "data\\dataset ABX Li v001_radii.csv")
