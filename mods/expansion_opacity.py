import pathlib
import warnings

import astropy.constants as c
import astropy.units as u
import numpy as np
import pandas as pd

try:
    import mods.opacity_mod as op
except:
    import opacity_mod as op

warnings.filterwarnings("ignore")

def Get_Opacity(full_atomic_data, atomic_info, T=5000, rho=1e-13,time=1,lambda_bin=10, line_binned=False, grid_bin='wave', grid_energy='Los Alamos',extension_name=''):
    (
        atomic_weights,
        ionization_energies,
        gfall_levels,
        gfall_lines,
        ground_levels,
        levels,
        lines,
    ) = full_atomic_data
    T=T*u.K
    rho = rho * u.g / u.cm ** 3
    time = time * u.day.to(u.s)
    atomic_number, ion_stages, type_calc = atomic_info
    exp_op=op.compute_expansion_opacity(atomic_number, ion_stages, lambda_bin,2400*10**(9), time, T, rho, ground_levels,gfall_levels,levels, atomic_weights, ionization_energies, lines, line_binned=line_binned)
    # exp_op.to_csv(f'exp_op_df_{type_calc}_T{T.value}.csv')
    save_dir = pathlib.Path("Opacities")
    save_dir.mkdir(exist_ok=True)
    if line_binned:
        columns = ["Photon Energy"] + [i for i in ion_stages]
        opacitydf=pd.DataFrame(columns=columns)
        type_exp='line_binned'
        save_dir2 = pathlib.Path(f'{save_dir}/{type_exp}')
        save_dir2.mkdir(exist_ok=True)
        lines['frequency'] =(c.c.to("cm/s")/(lines['wavelength'].values *u.AA).to(u.cm)).value
        if grid_energy=='Los Alamos':
            type_grid='Los Alamos'
            grid_energy=np.loadtxt('Grids/Grid_Energy_Losalamosbinned_0.00125_99.5.dat', dtype=float)
        elif grid_energy=='Default2000':
            type_grid='Default2000'
            grid_energy=np.logspace(-3,2,2000)
        elif grid_energy=='Default1000':
            type_grid='Default1000'
            grid_energy=np.logspace(-3,2,1000)
        elif grid_energy=='Default500':
            type_grid='Default500'
            grid_energy=np.logspace(-3,2,500)
        elif grid_energy=='Default100':
            type_grid='Default100'
            grid_energy=np.logspace(-3,2,100)
        elif grid_energy=='Default10000_lin':
            type_grid='Default10000_lin'
            grid_energy=np.linspace(0.001,100,10000)
        else:
            type_grid='Custom'
        for i in ion_stages:
            # print(ion_stages)
            # print('LINES')
            # print(lines.head())
            # print('EXP_OP')
            # print(exp_op.head())
            frequency=lines.loc[atomic_number,i,:]['frequency'].values
            energy=frequency*c.h.to("eV s").value
            opacity=exp_op.loc[atomic_number,i,:]['line_binned'].values
            # print('LINES_loc')
            # print(lines.loc[atomic_number,i,:].head())
            # print('EXP_OP_loc')
            # print(exp_op.loc[atomic_number,i,:].head())
            # minidf=pd.DataFrame({'wavelength':lines['wavelength'],'frequency':frequency,'energy':energy,'opacity':opacity,'f_lu':lines['f_lu'], 'state_density':lines['number_density']})
            # minidf.sort_values(by='energy',inplace=True)
            # minidf.to_csv(f'minidf_{type_calc}_{Tev:.2f}.csv')
            
            opacity_histo, edges = np.histogram(
            energy, bins=grid_energy, weights=opacity
            )
            grid_midpoints = 0.5 * (grid_energy[1:] + grid_energy[:-1])
            bin_width=np.array(np.array(grid_energy[1:]) - np.array(grid_energy[:-1]))/c.h.to("eV s").value
            opacity_histo = opacity_histo / bin_width
            opacitydf[i] = opacity_histo
        opacitydf['Photon Energy'] = grid_midpoints
        opacitydf.set_index ('Photon Energy', inplace=True)
        Tev=T.value/11604.52500617
        opacitydf.to_csv(f"{save_dir2}/{type_calc}_{atomic_number}_{Tev:.2f}_{type_grid}_lines.csv")
        if extension_name!='':
            opacitydf.to_csv(f"{save_dir2}/{type_calc}_{atomic_number}_{Tev:.2f}_{type_grid}_lines_{extension_name}.csv")

    else:
        columns = ["Wavelenght"] + [i for i in range(1, len(ion_stages) + 1)]
        opacitydf=pd.DataFrame(columns=columns)
        type_exp='expansion'
        save_dir2 = pathlib.Path(f'{save_dir}/{type_exp}')
        save_dir2.mkdir(exist_ok=True)
        if grid_bin=='wave':
            for i in ion_stages:
                opacity, grid_midpoints = op.make_expansion_opacity_grid(
                    exp_op.loc[atomic_number, i, :], 0, 25000, lambda_bin
                )
                opacitydf[i] = opacity
        elif grid_bin=='energy':
            for i in ion_stages:
                opacity, grid_midpoints = op.make_expansion_opacity_grid_energy(
                    exp_op.loc[atomic_number, i, :], 1e-2, 1e2, 1000
                )
                print(grid_midpoints)
                opacitydf[i] = opacity
        opacitydf["Wavelenght"] = grid_midpoints
        opacitydf.set_index("Wavelenght", inplace=True)
        opacitydf.to_csv(f"{save_dir2}/{type_calc}_{atomic_number}_{T.value}_{lambda_bin}.csv")
        if extension_name!='':
            opacitydf.to_csv(f"{save_dir2}/{type_calc}_{atomic_number}_{T.value}_{lambda_bin}_{extension_name}.csv")
    return opacitydf

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
    # print(full_atomic_data)
    # opacitydf = Get_Opacity(full_atomic_data, atomic_info, T=5800, line_binned=False)
    # # opacitydf = Get_Opacity(full_atomic_data, atomic_info, T=5000, line_binned=False)
    # print(opacitydf)