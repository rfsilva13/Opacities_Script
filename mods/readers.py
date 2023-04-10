import numpy as np
import pandas as pd

try:
    import mods.nist as nist
except:
    import nist as nist

# import modin.pandas as pd
try:
    from mods.auxiliary import data_optimize
except:
    from auxiliary import data_optimize

import re
import mendeleev as md
from fractions import Fraction
import roman


def J_PI_indexing(levels):
    """
    Assigns a unique J-PI index to each level in the input dataframe.

    Args:
    levels (pandas.DataFrame): Dataframe containing energy levels.

    Returns:
    pandas.DataFrame: Dataframe with an additional column for the J-PI index.
    """
    all_levels_temp = []
    for par in [0, 1]:
        levels_temp = levels[levels["p"] == par]
        all_levels_par_temp = []
        for j in np.unique(levels_temp["2j"]):
            # print(j, par)
            levels_j_temp = levels_temp[levels_temp["2j"] == j]
            # print(levels_j_temp)
            levels_j_temp = levels_j_temp.sort_values(by=["energy [cm-1]"])
            levels_j_temp["J_PI_index"] = [i for i in range(len(levels_j_temp))]
            # print('J_PI_index')
            # print(levels_j_temp)
            all_levels_par_temp.append(levels_j_temp)
            # print(all_levels_par_temp)
        all_levels_temp.append(pd.concat(all_levels_par_temp))
    levels = pd.concat(all_levels_temp)
    return levels


def read_levels_FAC(atom_number, ion_charge, file_levels):
    """
    Reads and optimizes energy levels from a csv file for a given atom and ion using the FAC method.

    Args:
        atom_number (int): The atomic number of the element.
        ion_charge (int): The charge state of the ion.
        file_levels (str): The path to the csv file containing the energy levels.

    Returns:
        pandas.DataFrame: A dataframe containing the optimized energy levels with columns for atomic number,
        ion charge, level index, energy in cm-1, parity, 2j quantum number, J/PI index,
        configuration label, level label, method used for calculation and priority (used for Tardis).
    """
    levels = pd.read_csv(
        file_levels,
        delimiter="\s+",
        names=[
            "level_index",
            "IBASE",
            "energy [cm-1]",
            "p",
            "VNL",
            "2j",
            "conf1",
            "configuration",
            "label",
        ],
        usecols=[0, 2, 3, 5, 7, 8],
        skiprows=12,
    )
    levels = data_optimize(levels)
    levels["2j"] = levels["2j"].astype(int)
    levels["energy [cm-1]"] = levels["energy [cm-1]"] * 8065.544004795713
    levels["atomic_number"] = atom_number
    levels["ion_charge"] = ion_charge
    levels["method"] = "FAC"
    levels["priority"] = 10
    levels = J_PI_indexing(levels)
    # levels["configuration"] = levels["configuration"].str.replace("", "")

### replace 1 but not 11, 10, 12, 13, 14, 15, 16, 17, 18, 19 regex
    levels["configuration"] = levels["configuration"].str.replace("(?<!\d)1(?!\d)", "")
    levels = levels[
        [
            "atomic_number",
            "ion_charge",
            "level_index",
            "energy [cm-1]",
            "p",
            "2j",
            "J_PI_index",
            "configuration",
            "label",
            "method",
            "priority",
        ]
    ]
    levels = data_optimize(levels)
    levels = levels.sort_values(by=["energy [cm-1]"])
    return levels


def read_transitions_FAC(atom_number, ion_charge, file_transitions):
    """
    Reads and optimizes the transitions data from a given file for a specific atom and ion.

    Args:
        atom_number (int): The atomic number of the element.
        ion_charge (int): The ion charge of the element.
        file_transitions (str): The path to the transition data file.

    Returns:
        pandas.DataFrame: A DataFrame containing the optimized transitions data for the given element.

    """
    transitions = pd.read_csv(
        file_transitions,
        delimiter="\s+",
        names=[
            "level_index_upper",
            "level_index_lower",
            "delta_energy",
            "gf",
            "tr_rate[1/s]",
            "monopole",
        ],
        skiprows=13,
        usecols=[0, 2, 4, 5, 6, 7],
    )

    transitions["atomic_number"] = atom_number
    transitions["ion_charge"] = ion_charge

    transitions["wavelength"] = 1e8 / (transitions["delta_energy"] * 8065.54429)
    transitions = transitions[
        [
            "atomic_number",
            "ion_charge",
            "level_index_lower",
            "level_index_upper",
            "wavelength",
            "gf",
            "tr_rate[1/s]",
            "monopole",
        ]
    ]
    transitions = data_optimize(transitions)
    return transitions


def get_energy_transitions(transitions, levels):
    """
    Get the energies of the lower and upper levels of a transition.

    Args:
    transitions (pandas.DataFrame): A DataFrame containing information about the transitions.
        Must include columns 'level_index_lower' and 'level_index_upper'.
    levels (pandas.DataFrame): A DataFrame containing information about the energy levels.
        Must include columns 'level_index' and 'energy [cm-1]'.

    Returns:
    tuple: Two pandas.Series containing the energies of the lower and upper levels of each transition, respectively.
    """
    energies_lower = transitions["level_index_lower"].map(
        levels.set_index("level_index")["energy [cm-1]"]
    )
    energies_upper = transitions["level_index_upper"].map(
        levels.set_index("level_index")["energy [cm-1]"]
    )
    return energies_lower, energies_upper

def get_J_PI_trans(transitions, levels):
    """
    Get the J_PI values for the lower and upper energy levels of a transition.

    Args:
        transitions (pandas.DataFrame): A DataFrame containing information about each transition, 
            including the index of the lower and upper energy levels.
        levels (pandas.DataFrame): A DataFrame containing information about each energy level, 
            including the J_PI value.

    Returns:
        tuple: A tuple containing two pandas.Series objects. The first Series contains the J_PI values 
            for the lower energy level of each transition. The second Series contains the J_PI values 
            for the upper energy level of each transition.
    """
    J_PI_lower = transitions["level_index_lower"].map(
        levels.set_index("level_index")["J_PI_index"]
    )
    J_PI_upper = transitions["level_index_upper"].map(
        levels.set_index("level_index")["J_PI_index"]
    )
    return J_PI_lower, J_PI_upper


def get_J_trans(transitions, levels):
    """
    Get the 2J quantum numbers for the lower and upper levels of a list of transitions.

    Args:
        transitions (pandas.DataFrame): DataFrame with columns 'level_index_lower' and 
            'level_index_upper' containing the indices of the lower and upper energy levels 
            for each transition.
        levels (pandas.DataFrame): DataFrame with columns 'level_index' and '2j' containing 
            the 2J quantum numbers for each energy level.

    Returns:
        tuple: Two pandas.Series containing the 2J quantum numbers for the lower and upper 
            energy levels of each transition, respectively.
    """
    jj_lower = transitions["level_index_lower"].map(
        levels.set_index("level_index")["2j"]
    )
    jj_upper = transitions["level_index_upper"].map(
        levels.set_index("level_index")["2j"]
    )
    return jj_lower, jj_upper


def get_P_trans(transitions, levels):
    """
    Calculates the transition probability for each lower and upper energy level.

    Args:
    transitions (pandas.DataFrame): A pandas DataFrame containing information about the energy levels and transition probabilities.
    levels (pandas.DataFrame): A pandas DataFrame containing information about the energy levels.

    Returns:
    tuple: Two pandas Series containing the transition probabilities for each lower and upper energy level.
    """
    P_lower = transitions["level_index_lower"].map(levels.set_index("level_index")["p"])
    P_upper = transitions["level_index_upper"].map(levels.set_index("level_index")["p"])
    return P_lower, P_upper


def get_all_info_transitions(transitions, levels):
    """
    Returns a dictionary containing all the possible information about the transitions.

    Parameters:
    transitions (dict): A dictionary containing information about the transitions.
    levels (dict): A dictionary containing information about the energy levels.

    Returns:
    dict: A dictionary containing all possible information about the transitions, including energy transitions,
          J_PI transitions, J transitions and P transitions.

    """
    transitions["energy_lower"], transitions["energy_upper"] = get_energy_transitions(
        transitions, levels
    )
    transitions["J_PI_lower"], transitions["J_PI_upper"] = get_J_PI_trans(
        transitions, levels
    )
    transitions["J_lower"], transitions["J_upper"] = get_J_trans(transitions, levels)
    transitions["P_lower"], transitions["P_upper"] = get_P_trans(transitions, levels)
    return transitions


def read_levels_transitions_FAC(atom_number, ion_charge, file_levels, file_transitions, indexing=False):
    """
    Reads the levels and transitions data from the given files for a specific atom and ion.

    Args:
        atom_number (int): The atomic number of the atom.
        ion_charge (int): The charge state of the ion.
        file_levels (str): The path to the file containing level information.
        file_transitions (str): The path to the file containing transition information.
        indexing (bool, optional): Whether to set indexes for levels and transitions. Defaults to False.

    Returns:
        tuple: A tuple containing two pandas dataframes: 
            levels: A dataframe with level information columns: 'level_index', 'energy', 'g', 'label'.
            transitions: A dataframe with transition information columns: 
                'level_index_lower', 'level_index_upper', 'wavelength', 'oscillator_strength'
    """
    levels = read_levels_FAC(atom_number, ion_charge, file_levels)
    transitions = read_transitions_FAC(atom_number, ion_charge, file_transitions)
    transitions = get_all_info_transitions(transitions, levels)
    if indexing:
        levels = levels.set_index(["atomic_number", "ion_charge", "level_index"])
        transitions = transitions.set_index(
            ["atomic_number", "ion_charge", "level_index_lower", "level_index_upper"]
        )
    return levels, transitions


def get_split_index(raw_levels):
    """
    This function takes a pandas DataFrame `raw_levels` containing a column named "2j" and returns an integer 
    representing the index where the levels in "2j" column start to differ. It first finds all rows with same 
    value for "2j" and then calculates the differences between their 
    indices. Finally, it returns the index of first row where a difference occurs.
    
    Get the index where the levels in 2j column of the raw_levels DataFrame start to differ.

    Args:
        raw_levels (pandas.DataFrame): The raw levels DataFrame containing 2j column.

    Returns:
        int: The index where the levels in 2j column start to differ.
    """
    js = np.where(raw_levels["2j"].values == raw_levels["2j"].values[0])
    distances = np.diff(js[0])
    split_index = js[0][np.where(distances != 1)[0][0] + 1]
    return split_index


def read_levels_HFR(atom_number, ion_charge, file_levels, indexing=False, parity_ground="NIST"):
    """
    Read atomic energy levels from a file in HFR format and optimize the data.

    Parameters:
        atom_number (int): Atomic number of the element.
        ion_charge (int): Ion charge of the element.
        file_levels (str): Path to the file containing the energy levels.
        indexing (bool, optional): If True, set the index of the dataframe to be multi-indexed by atomic number,
                                    ion charge and level index. Defaults to False.
        parity_ground (str or int, optional): Parity of the ground state. If "NIST", determine it using NIST ASD database.
                                                Defaults to "NIST".

    Returns:
        pandas.DataFrame: Dataframe containing information about atomic energy levels including optimized data.

    *Note: This function assumes that the energy levels in file_levels are in HFR format and that there is no header in
            the file. The columns must be separated by commas and named as "energy [cm-1]" and "2j".
            The function also assumes that J values are not explicitly defined in the input file.

    Examples:
        >>> levels = read_levels_HFR(6, 2, 'HFR_6_2_levels.csv')
    """
    if parity_ground == "NIST":
        parity_ground = check_parity_nist(atom_number, ion_charge)
    levels = pd.read_csv(file_levels, sep=",", names=["energy [cm-1]", "2j"])
    levels["energy [cm-1]"] = levels["energy [cm-1]"] * 1000
    levels["2j"] = (levels["2j"] * 2).astype(int)
    levels["level_index"] = levels.index
    split_index = get_split_index(levels)
    levels["p"] = [(parity_ground + 1) % 2] * split_index + [parity_ground] * (
        len(levels) - split_index
    )
    levels["atomic_number"] = atom_number
    levels["ion_charge"] = ion_charge
    levels["method"] = "HFR"
    levels["priority"] = 10
    levels = J_PI_indexing(levels)
    levels["configuration"] = ["-"] * len(levels)
    levels["label"] = ["-"] * len(levels)
    levels = levels[
        [
            "atomic_number",
            "ion_charge",
            "level_index",
            "energy [cm-1]",
            "p",
            "2j",
            "J_PI_index",
            "configuration",
            "label",
            "method",
            "priority",
        ]
    ]
    levels = levels.sort_values(by=["energy [cm-1]"])
    levels = data_optimize(levels)
    if indexing:
        levels = levels.set_index(["atomic_number", "ion_charge", "level_index"])
    return levels


def check_parity_nist(atom_number, ion_charge):
    """
    Returns the parity of the lowest energy level of a given atom and ion according to NIST database.

    Parameters:
    atom_number (int): The atomic number of the element.
    ion_charge (int): The charge state of the ion. For neutral atoms, use 0.

    Returns:
    str: Either 0 (even), 1 (odd).
    """
    levels = nist.GetNISTLevels(atom_number, ion_charge)
    parity = levels[levels["energy [cm-1]"] == 0]["p"].values[0]
    return parity

def parse_js(j):
    """
    Convert a string representation of a fraction to an integer, 
    by multiplying the fraction by 2 and rounding it to the nearest whole number.
    
    Args:
        j (str): A string representation of a fraction.
        
    Returns:
        An integer representation of the parsed fraction multiplied by 2. 
        Returns numpy.nan if an exception occurs during parsing.
    """
    try:
        return int(2*Fraction(j))
    except:
        return np.nan

def read_levels_paris(atom_number, ion_charge, indexing=False):
    """Reads energy levels of an ion using the Paris database.

    Args:
        atom_number (int): Atomic number of the element.
        ion_charge (int): Ionization state of the ion.
        indexing (bool, optional): Whether to set the index of the DataFrame 
            to atomic number, ion charge and level index. Defaults to False.

    Returns:
        pandas.DataFrame: DataFrame containing the energy levels of the ion.
            Columns include: 'atomic_number', 'ion_charge', 'level_index',
            'energy [cm-1]', 'p', '2j', 'J_PI_index', 'configuration',
            'label', 'method', and 'priority'.

    Note:
        The Paris database contains energy levels for ions in various elements.
        This function reads the tables using pandas.read_html() method from a 
        website and returns a DataFrame with columns containing information 
        about the level such as its energy, its parity, its total angular 
        momentum, etc. The levels are then optimized using data_optimize() function
        and returned as a pandas.DataFrame object.
    """
    element = md.element(atom_number)
    symbol = element.symbol
    element_name = element.name
    # Read even energy levels table

    df_even=pd.read_html(f"http://www.lac.universite-paris-saclay.fr/Data/Database/Tab-energy/{element_name}/{symbol}-tables/{symbol}{ion_charge+1}e.html", header=0)[0]
    # print(df_even.columns)
    try:
        df_even=df_even[['E(cm-1)', 'J']]
    except:
        df_even=df_even[['E (cm-1)', 'J']]
        df_even.rename(columns={'E (cm-1)': 'E(cm-1)'}, inplace=True)
    df_even['p']=0
    # Read odd energy levels table

    df_odd=pd.read_html(f"http://www.lac.universite-paris-saclay.fr/Data/Database/Tab-energy/{element_name}/{symbol}-tables/{symbol}{ion_charge+1}o.html", header=0)[0]
    try:
        df_odd=df_odd[['E(cm-1)', 'J']]
    except:
        df_odd=df_odd[['E (cm-1)', 'J']]
        df_odd.rename(columns={'E (cm-1)': 'E(cm-1)'}, inplace=True)
    df_odd['p']=1
    # Concatenate the tables and optimize the levels
    
    levels=pd.concat([df_even, df_odd], ignore_index=True)
    levels.rename(columns={'E(cm-1)': 'energy [cm-1]', 'J': '2j'}, inplace=True)
    levels['2j']=levels['2j'].apply(parse_js)
    levels.dropna(inplace=True)
    levels["energy [cm-1]"] = levels["energy [cm-1]"].apply(lambda x: x.replace(' ', ''))
    levels["energy [cm-1]"]=pd.to_numeric(levels["energy [cm-1]"], errors='coerce')
    levels.dropna(inplace=True)
    levels["level_index"] = levels.index
    levels["atomic_number"] = atom_number
    levels["ion_charge"] = ion_charge
    levels["method"] = "Paris"
    levels["label"] = ["-"] * len(levels)
    levels["configuration"] = ["-"] * len(levels)
    levels["priority"] = 10
    levels = J_PI_indexing(levels)
    levels = levels[
        [
            "atomic_number",
            "ion_charge",
            "level_index",
            "energy [cm-1]",
            "p",
            "2j",
            "J_PI_index",
            "configuration",
            "label",
            "method",
            "priority",
        ]
    ]
    levels = levels.sort_values(by=["energy [cm-1]"])
    levels = data_optimize(levels)
    if indexing:
        levels = levels.set_index(["atomic_number", "ion_charge", "level_index"])
    return levels


def read_transitions_DREAM(atom_number, ion_charge, file_transitions, indexing=False):
    """
    Reads in transition data from a DREAM-formatted file and returns levels and transitions dataframes.
    
    Parameters:
        atom_number (int): Atomic number of the element.
        ion_charge (int): Ion charge of the element.
        file_transitions (str): Path to the DREAM-formatted file containing transition information.
        indexing (bool): Optional. If True, adds an index column to the levels dataframe. Default is False.
        
    Returns:
        tuple: A tuple containing two pandas dataframes. The first dataframe contains level information 
               including energy, parity, and 2J values. The second dataframe contains transition information 
               including lower and upper level indices, wavelength, gf-value, gA-value, and CF-value.

    """
    data = pd.read_csv(
        file_transitions,
        sep="\s+",
        names=[
            "wavelength",
            "lower_energy",
            "lower_p",
            "lower_j",
            "upper_energy",
            "upper_p",
            "upper_j",
            "gf",
            "gA",
            "CF",
        ],
        skiprows=5,
        header=None,
        comment="*",
        skip_blank_lines=True,
        )
    data.dropna(inplace=True)
    data["lower_j"] = (data["lower_j"] * 2).astype(int)
    data['upper_j'] = (data['upper_j'] * 2).astype(int)
    data["lower_p"] = data["lower_p"].apply(lambda x: 1 if x == "(o)" else 0)
    data["upper_p"] = data["upper_p"].apply(lambda x: 1 if x == "(o)" else 0)
    levels = data[["lower_energy", "lower_p", "lower_j"]]
    levels.rename(
        columns={"lower_energy": "energy [cm-1]", "lower_p": "p", "lower_j": "2j"},
        inplace=True,
    )
    levels=levels.append(
        data[["upper_energy", "upper_p", "upper_j"]].rename(
            columns={"upper_energy": "energy [cm-1]", "upper_p": "p", "upper_j": "2j"}
        )
    )
    levels = levels.drop_duplicates()
    levels=levels.sort_values(by=["energy [cm-1]"], inplace=False)
    levels=levels.reset_index(drop=True)

    levels['level_index'] = levels.index
    levels['atomic_number'] = atom_number
    levels['ion_charge'] = ion_charge
    levels['method'] = 'DREAM'
    levels['priority'] = 10
    levels=levels[['atomic_number', 'ion_charge', 'level_index', 'energy [cm-1]', 'p', '2j','J_P', 'method', 'priority']]

    data["gf"] = 10**data["gf"]
    transitions = data[['wavelength', 'lower_energy', 'upper_energy', 'gf', 'gA', 'CF']]
    transitions['lower_level_index'] = transitions['lower_energy'].apply(lambda x: levels[levels['energy [cm-1]'] == x]['level_index'].values[0])
    transitions['upper_level_index'] = transitions['upper_energy'].apply(lambda x: levels[levels['energy [cm-1]'] == x]['level_index'].values[0])
    transitions['atomic_number'] = atom_number
    transitions['ion_charge'] = ion_charge
    transitions['method'] = 'DREAM'
    transitions['priority'] = 10
    transitions['label'] = '-'
    transitions = transitions[
        ['atomic_number', 'ion_charge', 'lower_level_index', 'upper_level_index', 'wavelength', 'gf', 'gA', 'CF', 'label', 'method', 'priority']
    ]
    return levels, transitions


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

def read_levels_transitions_Tanaka(atomic_number, ionic_charge, dir=".", indexing=False):
    """
    Read the levels and transitions from the Tanaka files.

    Parameters:
        atomic_number (int): the atomic number of the element of interest
        ionic_charge (int): the ionic charge of the ion of interest
        dir (str): the directory where the files are located (default is current directory)
        indexing (bool): whether to index dataframes by atomic number, ion charge, and level index or not (default is False)

    Returns:
        levelsdf (pandas.DataFrame): dataframe containing level information with columns 'atomic_number', 'ion_charge', 
                                     'level_index', 'configuration', 'energy [cm-1]', 'J_PI_index', 'p', '2j',
                                     'method', and 'priority'
        transitionsdf (pandas.DataFrame): dataframe containing transition information with columns 'atomic_number',
                                           'ion_charge', 'level_index_lower', 'level_index_upper', 
                                           'wavelength', and 'gf'
    """
    levels, transitions = get_files(atomic_number, ionic_charge+1, dir)
    levels=parse_levels(levels)
    transitions=parse_lines(transitions)
    ## Adding level info 
    levels["energy [cm-1]"] = levels["energy"] * 8065.544004795713
    levels["atomic_number"] = atomic_number
    levels["ion_charge"] = ionic_charge
    levels['2j']=(levels['g']-1)
    levels["label"] = None
    levels["method"] = "Tanaka"
    levels["priority"] = 10
    levels["configuration"] = levels["config"]
    levels.rename(columns={'P':'p'}, inplace=True)

    levels = J_PI_indexing(levels)
    levels=levels.sort_values(by=['energy [cm-1]'])
    levels=levels[['atomic_number', 'ion_charge', 'level_index', 'configuration', 'energy [cm-1]','J_PI_index', 'p', '2j', 'method', 'priority']]

    ## Adding transition info

    transitions["atomic_number"] = atomic_number
    transitions["ion_charge"] = ionic_charge
    transitions["energy_lower"] = transitions["level_index_lower"].map(
            levels.set_index("level_index")["energy [cm-1]"])
    transitions["energy_upper"] = transitions["level_index_upper"].map(
        levels.set_index("level_index")["energy [cm-1]"])

    transitions['2j_lower']=transitions['level_index_lower'].map(
        levels.set_index("level_index")["2j"])
    transitions['2j_upper']=transitions['level_index_upper'].map(
        levels.set_index("level_index")["2j"])
    transitions["wavelength"] = transitions["wavelength"] * 10.0

    transitions["gf"]=10**(transitions["log(g_lower*f)"])
    transitions.drop(columns=["log(g_lower*f)", "g_upper*A"], inplace=True)
    transitions = transitions.sort_index()

    if indexing:
        levels = levels.set_index(["atomic_number", "ion_charge", "level_index"])
        transitions = transitions.set_index(
        ["atomic_number", "ion_charge", "level_index_lower", "level_index_upper"]
    )
    return levels, transitions    

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

def read_atomic_data(data_dir,atomic_number, ion_charges, filename, type="FAC"):
    atomic_data = {}
    if type == "FAC":
        for ion_charge in ion_charges:
            #print("Reading {} for {} {}".format(type, atomic_number, ion_charge))
            generalname= f'{md.element(atomic_number).symbol}{(roman.toRoman(ion_charge+1))}_{filename}'
            file_levels = f"./{data_dir}/{generalname}.lev.asc"
            file_transitions = f"./{data_dir}/{generalname}.tr.asc"
            levels, transitions=read_levels_transitions_FAC(atomic_number, ion_charge, file_levels, file_transitions)


            atomic_data[ion_charge] = {"levels": levels, "transitions": transitions}

    # elif type == "MONS":
    #     for ion_charge in ion_charges:
    #         #print("Reading {} for {} {}".format(type, atomic_number, ion_charge))
    #         generalname= f'{mdl.element(atomic_number).symbol}{(roman.toRoman(ion_charge+1))}_{filename}'
    #         file_levels = f"./{data_dir}/{generalname}.lev"
    #         file_transitions = f"./{data_dir}/{generalname}.trans"
    #         levels, transitions=read_levels_transitions_MONS(atomic_number, ion_charge, file_levels, file_transitions)
    #         atomic_data[ion_charge] = {"levels": levels, "transitions": transitions}
    # elif type == "MONS_alt":
    #     for ion_charge in ion_charges:
    #         #print("Reading {} for {} {}".format(type, atomic_number, ion_charge))
    #         generalname= f'{mdl.element(atomic_number).symbol}{(roman.toRoman(ion_charge+1))}_{filename}'
    #         file_levels = f"./{data_dir}/{generalname}.lev_alt"
    #         file_transitions = f"./{data_dir}/{generalname}.trans_alt"
    #         levels, transitions=read_levels_transitions_MONS2(atomic_number, ion_charge, file_levels, file_transitions)
    #         atomic_data[ion_charge] = {"levels": levels, "transitions": transitions}
    # elif type == "Tanaka":
    #     for ion_charge in ion_charges:
    #         #print("Reading {} for {} {}".format(type, atomic_number, ion_charge))
    #         generalname= f'{mdl.element(atomic_number).symbol}{(roman.toRoman(ion_charge+1))}_{filename}'
    #         levels, transitions=read_levels_transitions_Tanaka(atomic_number, ion_charge, data_dir)
    #         atomic_data[ion_charge] = {"levels": levels, "transitions": transitions}
    # elif type == "Gaigalas":
    #     for ion_charge in ion_charges:
    #         #print("Reading {} for {} {}".format(type, atomic_number, ion_charge))
    #         generalname= f'{mdl.element(atomic_number).symbol}{(roman.toRoman(ion_charge+1))}_{filename}'
    #         levels=pd.read_parquet(f"./{data_dir}/{generalname}_levels.parquet")
    #         levels=levels[atom_number,ion_charge]['level_index']-1
    #         transitions=pd.read_parquet(f"./{data_dir}/{generalname}_transitions.parquet")
            
    #         atomic_data[ion_charge] = {"levels": levels, "transitions": transitions}
    else:
        print('No other type of atomic data is implemented yet')
    return atomic_data


if __name__ == "__main__":
    test_paris=read_levels_paris(90,1)
    print(test_paris['2j'].unique())