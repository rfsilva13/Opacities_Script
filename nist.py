import re
from fractions import Fraction

import mendeleev as me
import pandas as pd
import roman

try:
    from Readers import J_PI_indexing
except:
    from Readers import J_PI_indexing


def DownloadNistLevels(atomic_number, ionstage=0):
    element = me.element(atomic_number).symbol
    ion = element + " " + roman.toRoman(ionstage + 1)

    nist_URL = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl?de=0"
    ion_get = f'spectrum={ion.replace(" ", "+")}'
    # remaining_options = "&".join(
    #     [
    #         "submit=Retrieve",
    #         "units=0",
    #         "upper_limit=",
    #         "parity_limit=both",
    #         "conf_limit=All",
    #         "conf_limit_begin=",
    #         "conf_limit_end=",
    #         "term_limit=All",
    #         "term_limit_begin=",
    #         "term_limit_end=",
    #         "J_limit=",
    #         "format=3",
    #         "output=0",
    #         "page_size=15",
    #         "multiplet_ordered=0",
    #         "conf_out=on",
    #         "level_out=on",
    #         "term_out=on",
    #         "j_out=on",
    #         "perc_out=",
    #         "temp=",
    #     ]
    # )
    remaining_options = "&".join([
    'submit=Retrieve+Data',
    'units=0',
    'format=3',
    'output=0',
    'page_size=15',
    'multiplet_ordered=0',
    'conf_out=on',
    'term_out=on',
    'level_out=on',
    'j_out=on',
    'temp=']
    )
    full_URL = f"{nist_URL}&{ion_get}&{remaining_options}"

    df = pd.read_csv(full_URL, delimiter="\t+", header=0, engine="python")
    for col in df.columns:
        df[col] = df[col].str.replace(r'"', "")

    try:
        idx = df[df["Configuration"].str.contains("Nd")].index.values[0]
        df = df.iloc[:idx]
    except:
        ...

    return df


def parse_NIST_levels(atomic_number, ionstage, df):
    df["label"] = df["Configuration"]
    df["Configuration"] = (
        df["Configuration"]
        .apply(lambda x: re.sub(r"\([^)]*\)", "", x))
        .apply(lambda x: x.replace("..", "."))
        .apply(lambda x: x.replace(" ", ""))
        .apply(lambda x: x.strip("."))
    )
    df["J"] = df["J"].apply(lambda x: Fraction(x) * 2).astype(int)
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
    df = J_PI_indexing(df)
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
    return df


def GetNISTLevels(atomic_number, ionstage=0, number=1):
    dfs = []
    for i in range(number):
        dfs.append(
            parse_NIST_levels(
                atomic_number,
                ionstage + i,
                DownloadNistLevels(atomic_number, ionstage + i),
            )
        )
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    df = GetNISTLevels(atomic_number=60, ionstage=1, number=1)
    print(df)