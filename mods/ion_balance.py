import pathlib
import warnings

import astropy.constants as c
import astropy.units as u
import numpy as np
import pandas as pd
import roman
from tqdm import tqdm

try:
    import mods.opacity_mod as op
    import mods.readers as rd
except:
    import opacity_mod as op
    import readers as rd

warnings.filterwarnings("ignore")
def Get_Ionic_balance(full_atomic_data, atomic_info, T_min=1000, T_max=20000, T_sep=100, rho=1e-13, extension_name=""):
    (
        atomic_weights,
        ionization_energies,
        gfall_levels,
        gfall_lines,
        ground_levels,
        levels,
        lines,
    ) = full_atomic_data
    atomic_number, ion_stages, type_calc = atomic_info
    Ts = np.arange(T_min, T_max, T_sep) * u.K
    rho = rho * u.g / u.cm ** 3

    ions=[(atomic_number,i) for i in ion_stages]
    if 0 not in ion_stages:
        # print('0 not in ion_stages')
                #print('0 not in ion_stages')
        levels=levels.rename(columns={'energy [cm-1]':'energy'})
        levels['j']=levels['2j']*0.5
        levels=levels[['atomic_number', 'ion_charge', 'level_index', 'g', 'energy', 'label', 'method', 'priority', 'j']]
        levels.set_index(['atomic_number', 'ion_charge', 'level_index'], inplace=True)   
        gfall_ions = [(atomic_number, 0)]

        final_levels = ground_levels.drop(ions + gfall_ions)
        final_levels = pd.concat([final_levels, levels])
        #gfall = gfall_reader.levels.loc[atomic_number, 0, :]
        gfall=gfall_levels.loc[atomic_number, 0, :]
        gfall["atomic_number"] = atomic_number
        gfall["ion_charge"] = 0
        gfall["g"] = gfall["j"] * 2 + 1
        # print('levels')
        # print(levels)
        levels = pd.concat([final_levels, gfall])
        levels.sort_index()
    else:
        final_levels=ground_levels.drop(ions)
        levels=pd.concat([final_levels,levels])
        levels.sort_index()
    number_density=op.get_number_density(rho, atomic_number, atomic_weights)  

    ion_get=[ion_stages[0]-1]+ion_stages + [ion_stages[-1]+1] # get the ionization stages
    columns=['T']+[i for i in ion_get]
    ionizatio_df = pd.DataFrame(columns=ion_get)
    ionizatio_df["T"] = Ts
    ionizatio_df.set_index("T", inplace=True)
    for T in tqdm(Ts):
        n_ions, n_electron, partition_functions=op.compute_ionisation_balance(
        T, ionization_energies, levels, atomic_number, number_density)
        for i in ion_get:
            ionizatio_df[i][T.value] = n_ions[i]
        # ionizatio_df['Partition function'][T.value] = partition_functions
    save_dir = pathlib.Path(f'Ionization_balance')
    save_dir.mkdir(exist_ok=True)
    ionizatio_df.to_csv(f"{save_dir}/{type_calc}_{atomic_number}_{extension_name}.csv")
    return ionizatio_df

if __name__ == "__main__":
    atomic_number = 60
    dir_path = "Database/FAC_data"
    filename = "test"
    type_calc = "FAC"
    ion_stages = [1, 2]
    min_ion = "H"
    max_ion = "U"
    full_atomic_data, atomic_info = op.GetCompleteData(
        atomic_number, dir_path, filename, type_calc, ion_stages, min_ion, max_ion
    )
    print(full_atomic_data)
    ionizatio_df = Get_Ionic_balance(full_atomic_data, atomic_info)
    # # opacitydf = Get_Opacity(full_atomic_data, atomic_info, T=5000, line_binned=False)