import re
from fractions import Fraction
import numpy as np
import ssl
import mendeleev as me
import pandas as pd
import roman
import os

try:
    import mods.readers as rd
except:
    import readers as rd


def DownloadNistLevels(atomic_number: int, ionstage: int = 0) -> pd.DataFrame:
    """
    Downloads NIST energy levels data for a specific atomic number and ion stage.

    Args:
        atomic_number (int): Atomic number of the element.
        ionstage (int, optional): Ion stage of the element. Defaults to 0.

    Returns:
        pd.DataFrame: DataFrame containing the downloaded data.
    """
    element = me.element(
        atomic_number
    ).symbol  # Get the symbol of the element using its atomic number.
    ion = f"{element} {roman.toRoman(ionstage + 1)}"  # Get the ion name by adding Roman numeral representation of ion stage to element symbol.

    nist_URL = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl?de=0"
    ion_get = f'spectrum={ion.replace(" ", "+")}'  # Replace spaces with '+' in the ion name to create a valid URL parameter value.

    remaining_options = "&".join(
        [
            "submit=Retrieve+Data",
            "units=0",
            "format=3",
            "output=0",
            "page_size=15",
            "multiplet_ordered=0",
            "conf_out=on",
            "term_out=on",
            "level_out=on",
            "j_out=on",
            "temp=",
        ]
    )  # URL parameters for downloading the data.

    full_URL = (
        f"{nist_URL}&{ion_get}&{remaining_options}"  # Full URL for downloading data.
    )

    ssl._create_default_https_context = (
        ssl._create_unverified_context
    )  # Bypass SSL certificate verification.
    df = pd.read_csv(
        full_URL,
        delimiter="\t+",
        header=0,
        engine="python",
        comment=element,  # Ignore rows starting with the element symbol as a comment.
        names=["Configuration", "Term", "J", "Prefix", "Level (cm-1)", "Suffix"],
    )  # Read the downloaded CSV file into a pandas DataFrame.

    df = df.dropna()  # Drop rows with null values, if any.

    for col in df.columns:
        df[col] = df[col].str.replace(
            r'"', ""
        )  # Remove double quotes from all columns.

    try:
        idx = df[df["Configuration"].str.contains("Nd")].index.values[
            0
        ]  # Find the index of a row containing 'Nd' in the Configuration column, if present.
        # This is done to remove any extra levels that may be present due to poor formatting of NIST database.
        # A similar approach can be taken for other elements as well, depending on their unique configuration strings.
        # If no such row is found, all levels are kept as they are.
        # Note: This is specific to this use case and may not apply to all scenarios.
        df = df.iloc[:idx]  # Keep only the rows upto the index found above.
    except:
        ...

    return df


def parse_fraction(x):
    try:
        return int(Fraction(x) * 2)
    except:
        return np.nan


def parse_NIST_levels(atomic_number, ionstage, df):
    print("HERE")
    df["label"] = df["Configuration"]
    df["Configuration"] = (
        df["Configuration"]
        .apply(lambda x: re.sub(r"\([^)]*\)", "", x))
        .apply(lambda x: x.replace("..", "."))
        .apply(lambda x: x.replace(" ", ""))
        .apply(lambda x: x.strip("."))
        .apply(lambda x: re.sub(r"\<[^)]*\>", "", x))
        .apply(lambda x: x.replace("?", ""))
    )
    df["J"] = df["J"].apply(lambda x: parse_fraction(x))
    df = df.dropna()
    df["P"] = df["Term"].apply(lambda x: 1 if "*" in x else 0)
    df = df.rename(
        columns={
            "Configuration": "configuration",
            "J": "2j",
            "Level (cm-1)": "energy [cm-1]",
            "P": "p",
        }
    )
    df["energy [cm-1]"] = pd.to_numeric(df["energy [cm-1]"], errors="coerce")
    df = df.dropna(subset=["energy [cm-1]"])
    df["atomic_number"] = atomic_number
    df["ion_charge"] = ionstage
    df["level_index"] = df.index
    df["method"] = "NIST"
    df["priority"] = 10
    df["label"] = df["label"] + "." + df["Term"]
    # print(df)
    df = rd.J_PI_indexing(df)
    df = df[
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
    df = df.sort_values(by=["energy [cm-1]"])
    df = df.reset_index(drop=True)
    return df


def GetNISTLevels(atomic_number, ionstage=0, number=1):
    dfs = []
    path_use = "/home/rfsilva/FAC_optimizer/NIST_Database/"
    filename = f"{path_use}{atomic_number}_{ionstage}.csv"
    try:
        for i in range(number):
            dfs.append(pd.read_csv(filename))
    except:
        for i in range(number):
            dfs.append(
                parse_NIST_levels(
                    atomic_number,
                    ionstage + i,
                    DownloadNistLevels(atomic_number, ionstage + i),
                )
            )
    final_df = pd.concat(dfs, ignore_index=True)
    if not os.path.exists(f"{filename}"):
        final_df.to_csv(f"{filename}", index=False)
    return final_df


def DownloadNistTransitions(atomic_number, ionstage=0):
    element = me.element(atomic_number).symbol
    ion_stage = roman.toRoman(ionstage + 1)
    nist_URL = (
        f"https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra={element}+{ion_stage}+"
    )
    remaining_options = "&".join(
        [
            "limits_type=0",
            "low_w=",
            "upp_w=",
            "unit=0",
            "submit=Retrieve+Data",
            "de=0",
            "I_scale_type=1",
            "format=3",
            "line_out=1",
            "remove_js=on",
            "en_unit=0",
            "output=0",
            "bibrefs=1",
            "page_size=15",
            "show_obs_wl=1",
            "show_calc_wl=1",
            "unc_out=1",
            "order_out=0",
            "max_low_enrg=",
            "show_av=2",
            "max_upp_enrg=",
            "tsb_value=0",
            "min_str=",
            "A_out=0",
            "intens_out=on",
            "max_str=",
            "allowed_out=1",
            "forbid_out=1",
            "min_accur=",
            "min_intens=",
            "conf_out=on",
            "term_out=on",
            "enrg_out=on",
            "J_out=on",
        ]
    )
    full_URL = f"{nist_URL}&{remaining_options}"
    df = pd.read_csv(
        full_URL,
        delimiter="\t+",
        header=0,
        engine="python",
        comment=element,
        names=[
            "obs_wl_air(A)",
            "ritz_wl_air(A)",
            "intens",
            "Aki(s^-1)",
            "Acc",
            "Ei(cm-1)",
            "Ek(cm-1)",
            "conf_i",
            "term_i",
            "J_i",
            "conf_k",
            "term_k",
            "J_k",
            "Type",
            "tp_ref",
            "line_ref",
        ],
        usecols=[
            "obs_wl_air(A)",
            "ritz_wl_air(A)",
            "intens",
            "Aki(s^-1)",
            "Acc",
            "Ei(cm-1)",
            "Ek(cm-1)",
            "conf_i",
            "term_i",
            "J_i",
            "conf_k",
            "term_k",
            "J_k",
        ],
    )
    return df


def parse_NIST_transitions(atomic_number, ionstage, df):
    df["upper_label"] = df["conf_i"]
    df["lower_label"] = df["conf_k"]
    df["conf_i"] = (
        df["conf_i"]
        .apply(lambda x: re.sub(r"\([^)]*\)", "", x))
        .apply(lambda x: x.replace("..", "."))
        .apply(lambda x: x.replace(" ", ""))
        .apply(lambda x: x.strip("."))
        .apply(lambda x: re.sub(r"\<[^)]*\>", "", x))
        .apply(lambda x: x.replace("?", ""))
    )
    df["conf_k"] = (
        df["conf_k"]
        .apply(lambda x: re.sub(r"\([^)]*\)", "", x))
        .apply(lambda x: x.replace("..", "."))
        .apply(lambda x: x.replace(" ", ""))
        .apply(lambda x: x.strip("."))
        .apply(lambda x: re.sub(r"\<[^)]*\>", "", x))
        .apply(lambda x: x.replace("?", ""))
    )

    df["J_i"] = df["J_i"].apply(lambda x: parse_fraction(x.replace('"', "")))
    df["J_k"] = df["J_k"].apply(lambda x: parse_fraction(x.replace('"', "")))

    df = df.dropna()

    df["lower_p"] = df["term_i"].apply(lambda x: 1 if "*" in x else 0)
    df["upper_p"] = df["term_k"].apply(lambda x: 1 if "*" in x else 0)
    df = df.rename(
        columns={
            "conf_i": "lower_configuration",
            "conf_k": "upper_configuration",
            "J_i": "lower_2j",
            "J_k": "upper_2j",
            "Ei(cm-1)": "lower_energy [cm-1]",
            "Ek(cm-1)": "upper_energy [cm-1]",
        }
    )
    df["lower_energy [cm-1]"] = df["lower_energy [cm-1]"].apply(
        lambda x: x.replace('"', "")
    )
    df["upper_energy [cm-1]"] = df["upper_energy [cm-1]"].apply(
        lambda x: x.replace('"', "")
    )
    df["lower_energy [cm-1]"] = pd.to_numeric(
        df["lower_energy [cm-1]"], errors="coerce"
    )
    df["upper_energy [cm-1]"] = pd.to_numeric(
        df["upper_energy [cm-1]"], errors="coerce"
    )
    levels_NIST_tr = df[
        [
            "lower_label",
            "lower_configuration",
            "lower_energy [cm-1]",
            "lower_p",
            "lower_2j",
        ]
    ]
    levels_NIST_tr.rename(
        columns={
            "lower_label": "label",
            "lower_configuration": "configuration",
            "lower_energy [cm-1]": "energy [cm-1]",
            "lower_p": "p",
            "lower_2j": "2j",
        },
        inplace=True,
    )
    levels_NIST_tr = levels_NIST_tr.append(
        df[
            ["upper_configuration", "upper_energy [cm-1]", "upper_p", "upper_2j"]
        ].rename(
            columns={
                "upper_label": "label",
                "upper_configuration": "configuration",
                "upper_energy [cm-1]": "energy [cm-1]",
                "upper_p": "p",
                "upper_2j": "2j",
            }
        )
    )

    levels_NIST_tr = levels_NIST_tr.drop_duplicates(subset=["energy [cm-1]"])
    levels_NIST = GetNISTLevels(atomic_number, ionstage)
    levels_NIST["with_trans"] = levels_NIST["energy [cm-1]"].isin(
        levels_NIST_tr["energy [cm-1]"]
    )
    levels_NIST_filtered = levels_NIST[levels_NIST["with_trans"] == True]
    # print(levels_NIST_filtered)

    transitions_NIST_tr = df[
        [
            "obs_wl_air(A)",
            "ritz_wl_air(A)",
            "intens",
            "lower_energy [cm-1]",
            "upper_energy [cm-1]",
            "Aki(s^-1)",
            "Acc",
        ]
    ]
    transitions_NIST_tr = transitions_NIST_tr.rename(
        columns={
            "obs_wl_air(A)": "wavelength",
            "ritz_wl_air(A)": "wavelength_r",
            "intens": "intensity",
            "lower_energy [cm-1]": "lower_energy",
            "upper_energy [cm-1]": "upper_energy",
            "Aki(s^-1)": "A",
            "Acc": "Acc",
        },
    )
    transitions_NIST_tr["lower_level_index"] = transitions_NIST_tr[
        "lower_energy"
    ].apply(lambda x: try_energy_matching(x, levels_NIST_filtered))
    transitions_NIST_tr["upper_level_index"] = transitions_NIST_tr[
        "upper_energy"
    ].apply(lambda x: try_energy_matching(x, levels_NIST_filtered))
    transitions_NIST_tr["wavelength"] = transitions_NIST_tr["wavelength"].apply(
        lambda x: x.replace('"', "")
    )
    transitions_NIST_tr["wavelength_r"] = transitions_NIST_tr["wavelength_r"].apply(
        lambda x: x.replace('"', "")
    )
    transitions_NIST_tr["wavelength_r"] = transitions_NIST_tr["wavelength_r"].apply(
        lambda x: x.replace("+", "")
    )
    transitions_NIST_tr["wavelength"] = pd.to_numeric(
        transitions_NIST_tr["wavelength"], errors="coerce"
    )
    transitions_NIST_tr["wavelength_r"] = pd.to_numeric(
        transitions_NIST_tr["wavelength_r"], errors="coerce"
    )
    transitions_NIST_tr["intensity"] = transitions_NIST_tr["intensity"].apply(
        lambda x: x.replace('"', "")
    )
    transitions_NIST_tr["intensity"] = pd.to_numeric(
        transitions_NIST_tr["intensity"], errors="coerce"
    )
    transitions_NIST_tr["A"] = transitions_NIST_tr["A"].apply(
        lambda x: x.replace('"', "")
    )
    transitions_NIST_tr["A"] = pd.to_numeric(transitions_NIST_tr["A"], errors="coerce")
    transitions_NIST_tr["atomic_number"] = atomic_number
    transitions_NIST_tr["ion_charge"] = ionstage
    transitions_NIST_tr["method"] = "NIST_tr"
    transitions_NIST_tr["priority"] = 10
    transitions_NIST_tr["label"] = "-"
    transitions_NIST_tr = transitions_NIST_tr[
        [
            "atomic_number",
            "ion_charge",
            "lower_level_index",
            "upper_level_index",
            "wavelength",
            "wavelength_r",
            "intensity",
            "A",
            "Acc",
            "label",
            "method",
            "priority",
        ]
    ]

    return levels_NIST_filtered, transitions_NIST_tr


def try_energy_matching(energy, levels):
    """Try to match an energy to a level in the levels dataframe."""
    try:
        return int(levels[levels["energy [cm-1]"] == energy]["level_index"].values[0])
    except:
        return np.nan


def GetNISTtransitions(atomic_number, ionstage, number=1):
    levels = []
    transitions = []
    for i in range(number):
        levels_i, transitions_i = parse_NIST_transitions(
            atomic_number,
            ionstage + i,
            DownloadNistTransitions(atomic_number, ionstage + i),
        )
        levels.append(levels_i)
        transitions.append(transitions_i)
    levels = pd.concat(levels)
    transitions = pd.concat(transitions)
    return levels, transitions


if __name__ == "__main__":
    df = GetNISTLevels(atomic_number=66, ionstage=1, number=1)
    print(df)
    # df = DownloadNistTransitions(atomic_number=60, ionstage=1)
    # levels, transitions= GetNISTtransitions(atomic_number=60, ionstage=1)
    # print(levels)
    # print(transitions)
