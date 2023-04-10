import os
import numpy as np
import pandas as pd
import mendeleev as md
import re


class cd:
    """Context manager for changing the current working directory.

    Args:
        newPath (str): The path to the new working directory.

    Attributes:
        savedPath (str): The path to the original working directory.

    Examples:
        with cd('/path/to/new/directory'):
            # code to be executed in new directory

    """
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def data_optimize(df, object_option=False):
    """Reduce the size of the input dataframe
    Parameters
    ----------
    df: pd.DataFrame
        input DataFrame
    object_option : bool, default=False
        if true, try to convert object to category
    Returns
    -------
    df: pd.DataFrame
        data type optimized output dataframe
    """

    # loop columns in the dataframe to downcast the dtype
    for col in df.columns:
        # process the int columns
        if df[col].dtype == 'int':
            col_min = df[col].min()
            col_max = df[col].max()
            # if all are non-negative, change to uint
            if col_min >= 0:
                if col_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col]
            else:
                # if it has negative values, downcast based on the min and max
                if col_max < np.iinfo(np.int8).max and col_min > np.iinfo(np.int8).min:
                    df[col] = df[col].astype(np.int8)
                elif col_max < np.iinfo(np.int16).max and col_min > np.iinfo(np.int16).min:
                    df[col] = df[col].astype(np.int16)
                elif col_max < np.iinfo(np.int32).max and col_min > np.iinfo(np.int32).min:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col]
                    
        # process the float columns
        elif df[col].dtype == 'float':
            col_min = df[col].min()
            col_max = df[col].max()
            # downcast based on the min and max
            if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col]

        if object_option:
            if df[col].dtype == 'object':
                if len(df[col].value_counts()) < 0.5 * df.shape[0]:
                    df[col] = df[col].astype('category')

    return df


def find_neighbours(value, df, colname):
    """
    Find the nearest value to a given value in a dataframe column.

    Args:
    value (float): The value to find the nearest match for.
    df (pandas.DataFrame): The dataframe containing the column to search.
    colname (str): The name of the column to search.

    Returns:
    tuple: A tuple containing the index of the row with the nearest match, 
           the matched value, and the error percentage between them.
           If an exact match is found, error will be 0.

    Example:
    >>> df = pd.DataFrame({'A': [1.0, 2.5, 3.1]})
    >>> find_neighbours(2.0, df, 'A')
        (0, 1.0, 50.0)
    
    """
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        error = 0
        return int(exactmatch.index.values[0]), value, error
    else:
        energy=df.iloc[(df[colname]-value).abs().argsort()[:1]][colname].values[0]
        idxdf=df[df[colname] == energy].index.values[0]
        error = (abs(energy-value)/value)*100
        return int(idxdf), energy, error


def level_indentification(levels_1, levels_2, name_1='DREAM', name_2='FAC', create_dict=False):
    """
    This function creates a mapping table between two sets of quantum energy levels ('levels_1' and 'levels_2').
    The mapping is based on comparing the energies of each level in 'levels_1' to the nearest level in 'levels_2'.
    The output is a pandas DataFrame containing the corresponding index numbers of each mapped level from both sets,
    as well as an error metric indicating the difference between the matched energies.
    
    Args:
        levels_1 (pandas DataFrame): A table of quantum energy levels with columns '2j', 'p', and 'energy [cm-1]'.
        levels_2 (pandas DataFrame): Another table of quantum energy levels with the same columns as 'levels_1'.
        name_1 (str): A label for 'levels_1'. Default is 'DREAM'.
        name_2 (str): A label for 'levels_2'. Default is 'FAC'.
        create_dict (bool): If True, a dictionary mapping the index numbers from 'levels_1' to index numbers
                            in 'levels_2' will be returned alongside the DataFrame. Default is False.
                            
    Returns:
        pandas DataFrame: A table with columns ['2j', 'p', 'energy [cm-1]', '{name_2}_index',
                                               '{name_1}_index', and 'error'].
                                              
                          Each row represents a mapped pair of levels from both input tables.
                          The columns represent:
                            -'2j': The value of 2*j associated with this mapped energy level.
                            -'p': The value of p associated with this mapped energy level.
                            -'energy [cm-1]': The energy of the level in 'levels_1' that was mapped to 'levels_2'.
                            -'{name_2}_index': The index number in 'levels_2' of the level that was mapped to.
                            -'{name_1}_index': The index number in 'levels_1' of the level that was mapped from.
                            -'error': The absolute difference between the energies of the two mapped levels.
                            
        Optional: If create_dict is True, a dictionary is also returned. This dictionary maps each index number
                  from 'levels_1' to its corresponding index number in 'levels_2'.
    """
    levels_map=pd.DataFrame(columns=['2j', 'p', 'energy [cm-1]', f'{name_2}_index', f'{name_1}_index', 'error'])

    for jj in levels_1['2j'].unique():
        for p in levels_1['p'].unique():
            levels_2_filtered=levels_2[(levels_2['2j']==jj) & (levels_2['p']==p)]
            for energy in levels_1['energy [cm-1]'][(levels_1['2j']==jj) & (levels_1['p']==p)].unique():
                lev1_index=levels_1.index[(levels_1['2j']==jj) & (levels_1['p']==p) & (levels_1['energy [cm-1]']==energy)].values[0]
                lev2_index, energy, error=find_neighbours(energy, levels_2_filtered, 'energy [cm-1]')
                levels_2_filtered.drop(lev2_index, inplace=True)
                levels_map=levels_map.append({'2j':jj, 'p':p, 'energy [cm-1]':energy, f'{name_2}_index':int(lev2_index), f'{name_1}_index':int(lev1_index), 'error':error}, ignore_index=True)
    if create_dict:
        levels_map_dict=pd.Series(levels_map[f'{name_2}_index'].values,index=levels_map[f'{name_1}_index']).to_dict()
        return levels_map, levels_map_dict
    else:
        return levels_map


## make code from previous cell into a function
def get_number_after_pattern(string, pattern):
    """Return the number after a specified pattern in a given string.

    Args:
        string (str): A string to search for the pattern and extract the number.
        pattern (str): A pattern to search for in the string.

    Returns:
        str: The number found after the specified pattern.

    Raises:
        IndexError: If no number is found after the specified pattern.

    Example:
        >>> get_number_after_pattern('The price is $10', '$')
        '10'
    """
    return re.findall(r'{}(\d+)'.format(pattern), string)[0]


def parse_shell(string):
    """
    Parse a string containing a shell in machine-readable format into a LaTeX format.
    Args:
        string (str): A string containing a shell in machine-readable format.

    Returns:
        str: The shell in LaTeX format.

    Example:    
        >>> parse_shell('1s2')
        '1s^{2}'
    """
    pattern = r'(\w+)([+-])(\w+)\((\w+)\)'
    output_str = re.sub(pattern, r'(\1_\2^{\3})_{\4}', string)
    return output_str

def get_terms(string):
    """ This function takes a string and devide it by . and last number of each parcel into a list
    """
    shells = string.split(".")
    ## get all elements after a parenthesis into a list for each element in shells and remove those elements from shells
    terms = [re.findall(r'(\d+$)', shell)[0] for shell in shells]
    ## remove all elements AFTER a ) from shells    

    shells = [re.sub('(\d+$)', '', shell) for shell in shells]
    
    # # shells = [re.sub(r'\(([^)]+)\)', '', shell) for shell in shells]
    return shells, terms



### join the first two terms of an array in a string inside parenthesis

def join_first_two_terms(shells, terms):
    """ This function takes an array and join the first two terms of an array in a string inside parenthesis
    """
    if len(terms)==len(shells):
        terms = terms[1:]
    
    return "({})_{{{}}}".format("\,".join(shells[:2]), terms[0])




def parse_FAC_label(string):
    """ This function takes a string and parses it to a FAC label
    """
    shells, terms = get_terms(string)

    shells=[parse_shell(shell) for shell in shells]

    while len(shells)>1:
        shells[1]=join_first_two_terms(shells, terms)
        shells=shells[1:]
        terms=terms[1:]

    return shells[0]

def parse_nist_label_to_latex(label):
    shells=label.split('.')
    return '\,'.join([re.sub(r'(\d+$)', r'^{\1}', shell) for shell in shells])

