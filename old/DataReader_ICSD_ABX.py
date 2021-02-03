from chempy import Substance
import mendeleev as md

def readrawfile(filename):
    iterfile = iter(open(filename))
    columns = splitline(next(iterfile))
    formula_index = columns.index("StructuredFormula")
    structuretype_index = columns.index("StructureType")
    id_index = columns.index("CollectionCode")
    rawdata = []
    for line in iterfile:
        templist = splitline(line)
        tempcompnents = []
        tempcompnents.append(templist[formula_index])
        tempcompnents.append(templist[structuretype_index])
        tempcompnents.append(templist[id_index])
        rawdata.append(tempcompnents)
    return rawdata


def filterdata(rawdata):
    simple_compounds = []
    mixed_compounds = []
    for line in rawdata:
        formula = line[0]
        formula = formula.translate({ord(c): None for c in '()'})
        formula = formula.split(' ')
        if len(formula) == 3:
            try:
                temp = []
                temp.append(simple_formula_to_chemical(formula))
                temp.append(line[1])
                temp.append(line[2])
                simple_compounds.append(temp)
            except:
                print("Error, parse failed:", formula, line[2])
        elif len(formula) > 3:
            temp = []
            temp.append(formula)  # Here to modify
            temp.append(line[1])
            mixed_compounds.append(temp)
        elif len(formula) < 3:
            print("Error, ignored compound: ", formula, line[2])
    # labeling
    for line in simple_compounds:
        tempcompnents = []
        for i in line[0].composition:
            tempcompnents.append(int(i))
        line[0] = tempcompnents
        if "Perovskite" in line[1]:
            line[1] = 1
        else:
            line[1] = 0

    search_duplicates(simple_compounds)

    print()


def simple_formula_to_chemical(formula):
        chemical = Substance.from_formula(formula[0] + formula[1] + formula[2])
        return chemical


def splitline(line):
    templine = line.rstrip('\n')
    templist = templine.split('\t')
    return templist


def search_duplicates(compounds):
    chains = []
    unique = []
    dup = 0
    for i in range(len(compounds)):
        id_chain = []
        label = None
        for j in range(i+1, len(compounds)):
            if compounds[i][0] == compounds[j][0]:
                if len(id_chain) ==0:
                    id_chain.append(compounds[i][2])
                    label = compounds[i][1]
                id_chain.append(compounds[j][2])
                if not label==compounds[j][1]:
                    if not "Collision" in id_chain[0]:
                        formula = ""
                        for e in compounds[j][0]:
                            formula+=md.element(int(e)).symbol
                        id_chain.insert(0, f"Collision: {formula}")
        if(len(id_chain)) > 0:
            if not any(id_chain[1] in sublist for sublist in chains):
                chains.append(id_chain)
                if not "Collision" in id_chain[0]:
                    id_chain.insert(0, "Duplicate")
                dup += len(id_chain)-1
                print(id_chain)
        else:
            if not any(compounds[i][2] in sublist for sublist in chains):
                unique.append(compounds[i][0])
    print("****************************")
    col = 0
    for e in chains:
        if "Collision" in e[0]:
            col+=1
            print(e)
    print("****************************")
    print("Input size:", len(compounds))
    print("Unique compounds:" , len(unique)+len(chains))
    print("Duplicates: ")
    print("Duplicated records: ", dup)
    print(f"In {len(chains)} chains. {col} of them have label collisions")
    print("****************************")
    # todo something with collisions here







rawdata = readrawfile("data\\ABX3 F.Cl.Br.I -O.txt")
filterdata(rawdata)
