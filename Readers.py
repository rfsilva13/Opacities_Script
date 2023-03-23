import re

import mendeleev as mdl
import pandas as pd
import roman
import numpy as np

pd.set_option('display.max_columns', None)




def read_levels_transitions_FAC(
    atom_number,
    ion_charge,
    file_levels,
    file_transitions,
    only_levels=False,
):
    raw_levels = pd.read_csv(
        file_levels,
        delimiter="\s+",
        names=[
            "level_index",
            "IBASE",
            "energy",
            "parity",
            "VNL",
            "2J",
            "conf1",
            "conf2",
            "conf3",
        ],
        skiprows=12,
    )
    print(raw_levels)
    raw_levels["j"] = raw_levels["2J"] / 2.0
    levels = raw_levels[["level_index", "energy", "j"]].copy()
    levels["energy"] = levels["energy"] * 8065.544004795713
    levels["atomic_number"] = atom_number
    levels["ion_charge"] = ion_charge
    levels["label"] = None
    levels['P']=raw_levels['parity']
    levels["method"] = "FAC"
    levels["priority"] = 10
    levels["configuration"] = raw_levels["conf2"]

    if not only_levels:
        raw_transitions = pd.read_csv(
            file_transitions,
            delimiter="\s+",
            names=[
                "level_index_upper",
                "2j_upper",
                "level_index_lower",
                "2j_lower",
                "delta_energy",
                "gf",
                "TR_rate[1/s]",
                "Monopole",
            ],
            skiprows=13,
        )
        #print(raw_transitions.head())
        raw_transitions["j_lower"] = raw_transitions["2j_lower"].astype(float) / 2.0
        raw_transitions["j_upper"] = raw_transitions["2j_upper"].astype(float) / 2.0

        transitions = raw_transitions[
            ["level_index_upper", "j_upper", "level_index_lower", "j_lower", "gf"]
        ].copy()
        transitions["atomic_number"] = atom_number
        transitions["ion_charge"] = ion_charge
        transitions["energy_lower"] = transitions["level_index_lower"].map(
            levels.set_index("level_index")["energy"]
        )
        transitions["energy_upper"] = transitions["level_index_upper"].map(
            levels.set_index("level_index")["energy"]
        )
        transitions["wavelength"] = 1e8 / (
            transitions["energy_upper"] - transitions["energy_lower"]
        )
        transitions = transitions.set_index(
            ["atomic_number", "ion_charge", "level_index_lower", "level_index_upper"]
        )
        transitions = transitions.sort_index()
    levels = levels.set_index(["atomic_number", "ion_charge", "level_index"])

    if not only_levels:
        return levels, transitions
    return levels





def read_levels_transitions_MONS(
    atom_number, ion_charge, file_levels, file_transitions
):
    levels = pd.read_csv(file_levels, names=["energy", "j"])
    levels["energy"] = 1000 * levels["energy"]
    levels = levels.round({"energy": 0})
    levels = levels[["energy", "j"]]
    levels["j"] = levels["j"]
    levels = levels.sort_values(by="energy")
    levels = levels.drop_duplicates(subset="energy",)
    levels["atomic_number"] = atom_number
    levels["ion_charge"] = ion_charge
    levels["level_index"] = pd.RangeIndex(len(levels.index))
    levels["label"] = None
    levels["method"] = "MONS"
    levels["priority"] = 10

    columns = [
        "wavelength",
        "energy_lower",
        "P_lower",
        "j_lower",
        "energy_upper",
        "P_upper",
        "j_upper",
        "log gf",
        "gA",
        "CF",
    ]
    transitions = pd.read_csv(
        file_transitions, delimiter=r"\s+", skiprows=2, header=None
    )
    transitions.columns = columns
    transitions["atomic_number"] = atom_number
    transitions["ion_charge"] = ion_charge
    transitions["gf"] = 10 ** transitions["log gf"]
    transitions["wavelength"] = transitions["wavelength"]
    transitions["level_index_lower"] = transitions["energy_lower"].map(
        levels.set_index("energy")["level_index"]
    )
    transitions["level_index_upper"] = transitions["energy_upper"].map(
        levels.set_index("energy")["level_index"]
    )
    # transitions_temp=transitions.drop_duplicates(['level_index_upper'])
    # levels['P']=levels['level_index'].map(transitions_temp.set_index('level_index_upper')['P_upper'])    
    transitions = transitions[
        [
            "atomic_number",
            "ion_charge",
            "level_index_lower",
            "level_index_upper",
            "energy_lower",
            "energy_upper",
            "gf",
            "j_lower",
            "j_upper",
            "wavelength",
        ]
    ]
    transitions = transitions.dropna().set_index(
        ["atomic_number", "ion_charge", "level_index_lower", "level_index_upper"]
    )
    levels = levels.set_index(["atomic_number", "ion_charge", "level_index"])
    transitions = transitions.sort_index()
    print('LEVELS MONS')
    return levels, transitions


def read_atomic_data(data_dir,atomic_number, ion_charges, filename, type="FAC"):
    atomic_data = {}
    if type == "FAC":
        for ion_charge in ion_charges:
            #print("Reading {} for {} {}".format(type, atomic_number, ion_charge))
            generalname= f'{mdl.element(atomic_number).symbol}{(roman.toRoman(ion_charge+1))}_{filename}'
            file_levels = f"./{data_dir}/{generalname}.lev.asc"
            file_transitions = f"./{data_dir}/{generalname}.tr.asc"
            levels, transitions=read_levels_transitions_FAC(atomic_number, ion_charge, file_levels, file_transitions)
            atomic_data[ion_charge] = {"levels": levels, "transitions": transitions}
    elif type == "MONS":
        for ion_charge in ion_charges:
            #print("Reading {} for {} {}".format(type, atomic_number, ion_charge))
            generalname= f'{mdl.element(atomic_number).symbol}{(roman.toRoman(ion_charge+1))}_{filename}'
            file_levels = f"./{data_dir}/{generalname}.lev"
            file_transitions = f"./{data_dir}/{generalname}.trans"
            levels, transitions=read_levels_transitions_MONS(atomic_number, ion_charge, file_levels, file_transitions)
            atomic_data[ion_charge] = {"levels": levels, "transitions": transitions}
    elif type == "MONS_alt":
        for ion_charge in ion_charges:
            #print("Reading {} for {} {}".format(type, atomic_number, ion_charge))
            generalname= f'{mdl.element(atomic_number).symbol}{(roman.toRoman(ion_charge+1))}_{filename}'
            file_levels = f"./{data_dir}/{generalname}.lev_alt"
            file_transitions = f"./{data_dir}/{generalname}.trans_alt"
            levels, transitions=read_levels_transitions_MONS2(atomic_number, ion_charge, file_levels, file_transitions)
            atomic_data[ion_charge] = {"levels": levels, "transitions": transitions}
    elif type == "Tanaka":
        for ion_charge in ion_charges:
            #print("Reading {} for {} {}".format(type, atomic_number, ion_charge))
            generalname= f'{mdl.element(atomic_number).symbol}{(roman.toRoman(ion_charge+1))}_{filename}'
            levels, transitions=read_levels_transitions_Tanaka(atomic_number, ion_charge, data_dir)
            atomic_data[ion_charge] = {"levels": levels, "transitions": transitions}
    elif type == "Gaigalas":
        for ion_charge in ion_charges:
            #print("Reading {} for {} {}".format(type, atomic_number, ion_charge))
            generalname= f'{mdl.element(atomic_number).symbol}{(roman.toRoman(ion_charge+1))}_{filename}'
            levels=pd.read_parquet(f"./{data_dir}/{generalname}_levels.parquet")
            levels=levels[atom_number,ion_charge]['level_index']-1
            transitions=pd.read_parquet(f"./{data_dir}/{generalname}_transitions.parquet")
            
            atomic_data[ion_charge] = {"levels": levels, "transitions": transitions}
    else:
        print('No other type of atomic data is implemented yet')
    return atomic_data

def get_files(atomic_number, ionic_charge, dir="."):
    """
    Get the files for the given atomic number and ionic charge.
    """
    # Get the files
    files = (
        open(f"{dir}/{atomic_number}_{ionic_charge}.tnk", "r")
        .read()
        .split("# Transitions")
    )
    levels=files[0]
    transitions=files[1]
    return levels, transitions

def parity_to_number(parity):
    if parity=='even':
        return 0
    elif parity=='odd':
        return 1

def parse_levels(levels):
    levels_data=[] 
    for line in levels.splitlines()[9:]:
        config=line.split('{')[-1].split('}')[0].strip().split('  ')
        config=[''.join(i.split(' ')) for i in config]
        config='.'.join(config)
        splited=re.split('\s+',line)
        lines_array=[int(splited[1]), float(splited[2]), parity_to_number(splited[3]),float(splited[4]),str(config)]
        levels_data.append(lines_array)
    levelsdf=pd.DataFrame(levels_data,columns=['level_index','g','P','energy','config'])
    levelsdf['level_index']=levelsdf['level_index']-1
    return levelsdf

def parse_lines(lines):
    lines_data=[]
    for line in lines.splitlines()[2:]:
        line=re.split('\s+',line)[1:-1]
        lines_data.append(line)
    linesdf=pd.DataFrame(lines_data,columns=['level_index_upper','level_index_lower','wavelength','g_upper*A', 'log(g_lower*f)'], dtype=float)
    return linesdf

def read_levels_transitions_Tanaka(atomic_number, ionic_charge, dir="."):
    """
    Read the levels and transitions from the Tanaka files.
    """
    levels, transitions = get_files(atomic_number, ionic_charge+1, dir)
    levels=parse_levels(levels)
    transitions=parse_lines(transitions)
    ## Adding level info 
    levels["energy"] = levels["energy"] * 8065.544004795713
    levels["atomic_number"] = atomic_number
    levels["ion_charge"] = ionic_charge
    levels['j']=(levels['g']-1)/2.0
    levels["label"] = None
    levels["method"] = "Tanaka"
    levels["priority"] = 10
    levels=levels.sort_values(by=['energy'])
    ## Adding transition info

    transitions["atomic_number"] = atomic_number
    transitions["ion_charge"] = ionic_charge
    transitions["energy_lower"] = transitions["level_index_lower"].map(
            levels.set_index("level_index")["energy"])
    transitions["energy_upper"] = transitions["level_index_upper"].map(
        levels.set_index("level_index")["energy"])

    transitions['j_lower']=transitions['level_index_lower'].map(
        levels.set_index("level_index")["j"])
    transitions['j_upper']=transitions['level_index_upper'].map(
        levels.set_index("level_index")["j"])
    transitions["wavelength"] = transitions["wavelength"] * 10.0
    transitions = transitions.set_index(
            ["atomic_number", "ion_charge", "level_index_lower", "level_index_upper"]
        )
    transitions["gf"]=10**(transitions["log(g_lower*f)"])
    transitions.drop(columns=["log(g_lower*f)", "g_upper*A"], inplace=True)
    transitions = transitions.sort_index()
    levels = levels.set_index(["atomic_number", "ion_charge", "level_index"])

    return levels, transitions    


from bisect import bisect_left

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def get_level_index(levels, energy):
    """
    Returns the index of the level closest to the energy
    """
    return levels.index[levels["energy"] == take_closest(np.array(levels["energy"]), energy)].tolist()[0]




def read_levels_transitions_MONS2(
    atom_number, ion_charge, file_levels, file_transitions
):
    levels = pd.read_csv(file_levels, names=["energy", "j"])
    levels = levels[["energy", "j"]]
    levels["j"] = levels["j"]
    levels = levels.sort_values(by="energy")
    levels = levels.drop_duplicates(subset="energy",)
    levels["atomic_number"] = atom_number
    levels["ion_charge"] = ion_charge
    levels["level_index"] = pd.RangeIndex(len(levels.index))
    levels["label"] = None
    levels["method"] = "MONS"
    levels["priority"] = 10
    transitions = pd.read_csv(
        file_transitions, names=["wavelength","energy_lower","gf",],sep=',')
    #print('File read')
    #print(transitions.head())
    transitions_unique = transitions.drop_duplicates(subset=["energy_lower"])
    transitions_unique = transitions_unique.sort_values(by="energy_lower")
    # transitions_unique["level_index_lower"] = transitions_unique.apply(lambda row: get_level_index(levels, row["energy_lower"]), axis=1)   
    temp_trans=transitions_unique.copy()
    # print(transitions_unique)
    # print(transitions_unique.loc[1].index.to_list())
    # print(temp_trans.loc[1].index.to_list())
    for id,trans in transitions_unique.iterrows():
        # print(id)
        temp=temp_trans.loc[id]
        print(temp)
        idx=get_level_index(levels, temp['energy_lower'])
        print(idx)
        temp_trans['level_index_lower']=idx
        temp_trans.drop(index=id, inplace=True)  

    transitions=pd.merge(transitions,transitions_unique[["level_index_lower","energy_lower"]],on=["energy_lower"])
    transitions["atomic_number"] = atom_number
    transitions["ion_charge"] = ion_charge

    #print('Working on mapping js')
    transitions['j_lower']=transitions['level_index_lower'].map(
        levels.set_index("level_index")["j"])

    #print('Mapping done')
    transitions["level_index_upper"] = ['No info' for i in range(len(transitions))]
    transitions["j_upper"] = ['No info' for i in range(len(transitions))]
    transitions["energy_upper"] = ['No info' for i in range(len(transitions))]
    #print('transitions')
    #print(transitions.head())
    transitions = transitions[
        [
            "atomic_number",
            "ion_charge",
            "level_index_lower",
            "level_index_upper",
            "energy_lower",
            "energy_upper",
            "gf",
            "j_lower",
            "j_upper",
            "wavelength",
        ]
    ]
    #print('transitions')
    #print(transitions.head())
    transitions = transitions.dropna().set_index(
        ["atomic_number", "ion_charge","level_index_lower", "level_index_upper"]
    )
    #print(transitions)
    levels = levels.set_index(["atomic_number", "ion_charge", "level_index"])
    # transitions = transitions.sort_index()
    levels["energy"] = 1000 * levels["energy"]
    transitions["energy_lower"] = 1000 * transitions["energy_lower"]
    return levels, transitions



if __name__ == "__main__":
    levels_file='Database/MONS_data/NdII_best.lev_alt'
    transitions_file='Database/MONS_data/NdII_best.trans_alt'
    atom_number=60
    ion_charge=1
    levels, transitions= read_levels_transitions_MONS2(atom_number, ion_charge, levels_file, transitions_file)

    #print(levels)
    #print(transitions)
    
    lines=pd.read_csv('MONS_data/NdII_best.trans_alt',names=["wavelength","energy_lower","gf",],sep=',')
    print(lines)

