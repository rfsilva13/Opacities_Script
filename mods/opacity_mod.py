from pathlib import Path

import astropy.constants as c
import astropy.units as u
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import mods.readers as rd
except:
    import readers as rd

from carsus.io.kurucz import GFALLReader
from carsus.io.nist import NISTIonizationEnergies, NISTWeightsComp

def get_ground_levels(ionization_energies, atom_number):
    ground_levels = ionization_energies.get_ground_levels()
    ground_levels["level_index"] = 0
    ground_levels = ground_levels.set_index(
        ["atomic_number", "ion_charge", "level_index"]
    )
    ground = ground_levels.loc[atom_number, :, :]
    ground["label"] = None
    ground["method"] = "nist"
    ground["priority"] = 10
    ground["j"] = 0.5 * (ground["g"] - 1)
    return ground

def get_general_data(min_ion,max_ion, atom_number):
    #Get the weights and ionization energies from NIST
    extent=f'{min_ion}-{max_ion}'
    # try:
    #     Path('./Database/GeneralData/').mkdir()

    atomic_weights = NISTWeightsComp()
    ionization_energies = NISTIonizationEnergies(extent)
    gfall_reader = GFALLReader(extent)
    gfall_reader.levels.to_parquet(f'./Database/GeneralData/{extent}.levelsgfall')
    gfall_reader.lines.to_parquet(f'./Database/GeneralData/{extent}.linesgfall')
    atomic_weights.base.to_parquet(f'./Database/GeneralData/{extent}.weights')
    ground_levels = get_ground_levels(ionization_energies, atom_number)
    ionization_energies.base.to_frame().to_parquet(f'./Database/GeneralData/{extent}.ionenergies')
    ground_levels.to_parquet(f'./Database/GeneralData/{atom_number}.groundlevels')
    gfall_levels=gfall_reader.levels
    gfall_lines=gfall_reader.lines
    # except FileExistsError:
    #     #print('Using existing general data')
    #     atomic_weights=pd.read_parquet(f'./Database/GeneralData/{extent}.weights')
    #     ionization_energies=pd.read_parquet(f'./Database/GeneralData/{extent}.ionenergies')
    #     gfall_levels=pq.ParquetDataset(f'./Database/GeneralData/{extent}.levelsgfall').read().to_pandas()
    #     gfall_lines=pq.ParquetDataset(f'./Database/GeneralData/{extent}.linesgfall').read().to_pandas()
    #     ground_levels=pd.read_parquet(f'./Database/GeneralData/{atom_number}.groundlevels')
    return atomic_weights,ionization_energies,gfall_levels,gfall_lines,ground_levels

def filter_configs(atomic_data,ion_charge,configs):
    """
    Filter the atomic data to only include the configurations specified in configs.
    """
    levels=atomic_data[ion_charge]['levels']
    transitions=atomic_data[ion_charge]['transitions']
    levels_to_use=levels[levels['configuration'].isin(configs)]
    indexes=levels_to_use.index.get_level_values("level_index")
    transitions_filt=transitions[transitions.index.isin(indexes, level='level_index_lower')]
    transitions_filt=transitions_filt[transitions_filt.index.isin(indexes, level='level_index_upper')]
    atomic_data[ion_charge]['levels']=levels_to_use
    atomic_data[ion_charge]['transitions']=transitions_filt
    return atomic_data

def final_data(atomic_data):
    levelsfinal=pd.concat([atomic_data[i]['levels'] for i in atomic_data.keys()])
    transitionsfinal=pd.concat([atomic_data[i]['transitions'] for i in atomic_data.keys()])
    # print('levelsfinal')
    # print(levelsfinal)
    # print('transitionsfinal')
    # print(transitionsfinal)
    levelsfinal['g'] = levelsfinal['2j'] + 1
    transitionsfinal['f_lu'] = transitionsfinal['gf'] / (transitionsfinal['J_lower']*2+1)
    return levelsfinal,transitionsfinal

def GetCompleteData(
    atom_number,
    dir_path,
    filename,
    type_calc,
    ion_stages=[1, 2],
    min_ion="H",
    max_ion="U",
    nConfig=('all', 'all')
):
    (
        atomic_weights,
        ionization_energies,
        gfall_levels,
        gfall_lines,
        ground_levels,
    ) = get_general_data(min_ion, max_ion, atom_number)
    atomic_data = rd.read_atomic_data(
        dir_path, atom_number, ion_stages, filename, type_calc
    )
    #print('atomic_data::')
    #print(atomic_data)
    if type_calc == "FAC":
        #print("As we working with FAC data - Filtering by configuration due to memory issues")
        for k, ion_charge in enumerate(ion_stages):
            nConfig_ion = nConfig[k]
            if isinstance(nConfig_ion, int):
                configs=list(atomic_data[ion_charge]['levels']['configuration'].unique())
                configs=configs[:nConfig_ion]
                atomic_data=filter_configs(atomic_data,ion_charge,configs)
            elif isinstance(nConfig_ion, list):
                configs=nConfig_ion
                atomic_data=filter_configs(atomic_data,ion_charge,configs)
    #print('before final')
    levels, lines = final_data(atomic_data)
    #print('after final')
    #print(levels)
    #print(lines)
    atomic_info = [atom_number, ion_stages, type_calc]
    full_atomic_data = [
        atomic_weights,
        ionization_energies,
        gfall_levels,
        gfall_lines,
        ground_levels,
        levels,
        lines,
    ]
    return full_atomic_data, atomic_info

def filter_configs(atomic_data,ion_charge,configs):
    """
    Filter the atomic data to only include the configurations specified in configs.
    """
    levels=atomic_data[ion_charge]['levels']
    transitions=atomic_data[ion_charge]['transitions']
    levels_to_use=levels[levels['configuration'].isin(configs)]
    indexes=levels_to_use.index.get_level_values("level_index")
    transitions_filt=transitions[transitions.index.isin(indexes, level='level_index_lower')]
    transitions_filt=transitions_filt[transitions_filt.index.isin(indexes, level='level_index_upper')]
    atomic_data[ion_charge]['levels']=levels_to_use
    atomic_data[ion_charge]['transitions']=transitions_filt
    return atomic_data

def get_number_density(rho, atom_number, weights):
    #M = weights.loc[atom_number]
    M=weights.loc[atom_number]['mass']
    number_density = c.N_A / (M * u.g / u.mol) * rho
    return number_density

##@nb.jit
def make_phis(T, ionisation_energies, partition_functions):
    beta = 1.0 / (c.k_B * T)
    g_electron = compute_g_electron(beta)
    ion_energies = (ionisation_energies.values * u.eV).to(u.J)
    phi = (
        2
        * g_electron
        * np.exp(-ion_energies * beta)
        * np.array(partition_functions[1:])
        / np.array(partition_functions[:-1])
    )
    return phi


def compute_ionisation_balance(
    T, ionisation_energies, levels, atom_number, number_density
):
    partition_functions = make_all_partition_functions(levels, atom_number, T)
    phis = make_phis(T, ionisation_energies.loc[atom_number]['ionization_energy'], partition_functions).value
    n_ions, n_electron = calculate_Saha_LTE(phis, partition_functions, number_density)
    return n_ions, n_electron, partition_functions


def make_all_partition_functions(levels, atom_number, T):
    partition_functions = []
    for ionstage in range(atom_number):
        level_energies = levels.loc[atom_number, ionstage, :]
        level_energies = (
            level_energies["energy"].values / 8065.544004795713 * u.eV
        ).to(u.J)
        statistical_weights = levels.loc[atom_number, ionstage, :]
        statistical_weights = statistical_weights["g"].values
        beta = 1 / (c.k_B * T)
        partition_function = (
            statistical_weights * np.exp(-level_energies * beta)
        ).sum()
        partition_functions.append(partition_function.value)
    partition_functions.append(1)
    return partition_functions

##@nb.jit
def calculate_with_n_electron(phi, partition_function, n_electron, number_density):
    phi_electron = np.nan_to_num(phi / n_electron)
    # import pdb; pdb.set_trace()
    phis_product = np.cumprod(phi_electron, 0)
    tmp_ion_populations = np.empty((phi_electron.shape[0] + 1))
    tmp_ion_populations[0] = number_density / (1 + np.sum(phis_product, axis=0))
    tmp_ion_populations[1:] = tmp_ion_populations[0] * phis_product
    tmp_ion_populations[tmp_ion_populations < 1e-20] = 0.0
    return tmp_ion_populations

##@nb.jit
def compute_g_electron(beta):
    return (
        (2 * np.pi * c.m_e.cgs.value / beta.cgs.value) / (c.h.cgs.value ** 2)
    ) ** 1.5


def calculate_Saha_LTE(phi, partition_function, number_density):
    n_e_convergence_threshold = 1e-4
    n_electron_iterations = 0
    number_density = number_density.to(1 / u.cm ** 3).value
    n_electron = 1e-6 * number_density
    while True:

        ion_number_density = calculate_with_n_electron(
            phi, partition_function, n_electron, number_density
        )
        ion_numbers = np.arange(ion_number_density.shape[0])
        new_n_electron = (ion_number_density * ion_numbers).sum(axis=0)
        n_electron_iterations += 1
        if n_electron_iterations > 100:
            print(f"n_electron iterations above 100 ({n_electron_iterations})")
        if np.all(
            np.abs(new_n_electron - n_electron) / n_electron < n_e_convergence_threshold
        ):
            break
        n_electron = 0.5 * (new_n_electron + n_electron)

    return ion_number_density / number_density, n_electron

##@nb.jit
def compute_level_population_fraction(partition_function, energies, g_factors, T):
    beta = -energies / (c.k_B * T)
    level_frac = g_factors / g_factors[0] * np.exp(beta)
    normalised_level_frac = level_frac / np.sum(level_frac)
    return normalised_level_frac

##@nb.jit
def compute_tau_sobolev(number_density, level_pop_frac, ion_frac, lines, time):
    state_density = number_density * ion_frac * level_pop_frac
    prefactor = np.pi * c.e.gauss ** 2 / (c.m_e.cgs * c.c.cgs)
    wavelengths = (lines["wavelength"].values * u.AA).to(u.cm)
    oscillator_strengths = lines["f_lu"].values
    tau_sobolev = prefactor * oscillator_strengths * time * state_density * wavelengths
    return tau_sobolev

##@nb.jit
def compute_linebinned_opacities(frequency_bin, lines, rho, weights,number_density, level_pop_frac, ion_frac):
    state_density = number_density * ion_frac * level_pop_frac
    # #print(c.c.to("cm/s"))
    # #print((lines['wavelength']*u.AA).to(u.cm))
    lines['frequency'] =(c.c.to("cm/s")/(lines['wavelength'].values *u.AA).to(u.cm)).value
    lines['photon_energy']=lines['frequency']*c.h.to("eV s").value
    # #print(lines['frequency'])
    prefactor = np.pi * c.e.gauss ** 2 / (c.m_e.cgs * c.c.cgs *rho)
    line_binned_opacity=prefactor*lines['f_lu']*state_density
    return line_binned_opacity
##@nb.jit
def compute_expansion_opacities(lambda_bin, lines, rho, time, tau_sobolev):
    wave_bin = (lambda_bin * u.AA).to(u.cm)
    wavelengths = (lines["wavelength"].values * u.AA).to(u.cm)
    expansion_opacities = (
        1
        / (rho * c.c.to("cm/s") * time)
        * wavelengths
        / wave_bin
        * (1 - np.exp(-tau_sobolev))
    )
    return expansion_opacities

# @profile
def make_expansion_opacity_df(
    atom_number,
    ion_charges,
    T,
    n_ions,
    levels_data,
    lines_data,
    time,
    rho,
    lambda_bin,
    frequency_bin,
    partition_functions,
    weights,
    line_binned=False
):
    lines_data=lines_data.set_index(['atomic_number','ion_charge','level_index_lower','level_index_upper'])
    dfs=[]
    for i in ion_charges:
        levels = levels_data.loc[atom_number, i, :]
        energies = (levels["energy"].values / 8065.54429 * u.eV).to(u.J)
        g_factors = levels["g"]
        lines = lines_data.loc[atom_number, i, :]
        partition_function = partition_functions[i]
        lower_levels = lines.index.get_level_values("level_index_lower")
        number_density = get_number_density(rho, atom_number, weights)
        ion_frac = n_ions[i]
        levels["level_pop_frac"] = compute_level_population_fraction(
            partition_function, energies, g_factors, T
        )
        # level_pop_frac = 
        level_pop_frac = levels.loc[lower_levels]["level_pop_frac"].values
        lines["level_pop_frac"] = level_pop_frac
        lines["number_density"] = number_density * ion_frac * level_pop_frac
        tau_sobolev = compute_tau_sobolev(
            number_density, level_pop_frac, ion_frac, lines, time
        )
        lines["tau_sobolev"] = tau_sobolev
        lines = lines.sort_values(["wavelength"])
        if line_binned:
            lines["line_binned"] =compute_linebinned_opacities(frequency_bin, lines, rho, weights,number_density, level_pop_frac, ion_frac)
        else:
            lines["expansion_opacity"] = compute_expansion_opacities(
                lambda_bin, lines, rho, time, lines["tau_sobolev"].values
            )
    #     if i == 0:
    #         singly = lines.copy()
    #         singly['atomic_number'] = atom_number
    #         singly['ion_charge'] = i+1
    #         singly = singly.reset_index().set_index(['atomic_number', 'ion_charge','level_index_lower','level_index_upper'])
    #     if i == 1:
    #         doubly = lines.copy()
    #         doubly['atomic_number'] = atom_number
    #         doubly['ion_charge'] = i+1
    #         doubly = doubly.reset_index().set_index(['atomic_number', 'ion_charge','level_index_lower','level_index_upper'])
    # opacity_df = pd.concat([singly, doubly])
    # opacity_df=data_optimize(opacity_df)
        lines['atomic_number'] = atom_number
        lines['ion_charge'] = i
        lines = lines.reset_index().set_index(['atomic_number', 'ion_charge','level_index_lower','level_index_upper'])
        dfs.append(lines)
    opacity_df = pd.concat(dfs)
    return opacity_df

# @profile
def compute_expansion_opacity(
    atom_number,
    ion_charges,
    lambda_bin,
    frequency_bin,
    time,
    T,
    rho,
    ground_levels,
    gfall_levels,
    levels,
    atomic_weights,
    ionisation_energies,
    transitions,
    line_binned=False,
):
    ions=[(atom_number,i) for i in ion_charges]
    if 0 not in ion_charges:
        #print('0 not in ion_stages')
        levels=levels.rename(columns={'energy [cm-1]':'energy'})
        levels['j']=levels['2j']*0.5
        levels=levels[['atomic_number', 'ion_charge', 'level_index', 'g', 'energy', 'label', 'method', 'priority', 'j']]
        levels.set_index(['atomic_number', 'ion_charge', 'level_index'], inplace=True)    
        gfall_ions = [(atom_number, 0)]

        final_levels = ground_levels.drop(ions + gfall_ions)
        # print('final_levels')
        # print(final_levels)
        # print('levels')
        # print(levels)
        final_levels = pd.concat([final_levels, levels])
        #gfall = gfall_reader.levels.loc[atomic_number, 0, :]
        gfall=gfall_levels.loc[atom_number, 0, :]
        gfall["atomic_number"] = atom_number
        gfall["ion_charge"] = 0
        gfall["g"] = gfall["j"] * 2 + 1
        # print('final_levels',final_levels)
        # print('gfall',gfall)
        levels = pd.concat([final_levels, gfall])
        # print('levels',levels)
        levels.sort_index()
    else:
        final_levels=ground_levels.drop(ions)
        levels=pd.concat([final_levels,levels])
        levels.sort_index()

    number_density = get_number_density(rho, atom_number, atomic_weights)
    n_ions, n_electron, partition_functions = compute_ionisation_balance(
        T, ionisation_energies, levels, atom_number, number_density
    )
    exp_op_doubly = make_expansion_opacity_df(
        atom_number,
        ion_charges,
        T,
        n_ions,
        levels,
        transitions,
        time,
        rho,
        lambda_bin,
        frequency_bin,
        partition_functions,
        atomic_weights,
        line_binned=line_binned,
    )
    return exp_op_doubly

def make_expansion_opacity_grid(ion_df, lambda_min, lambda_max, lambda_bin):
    wavelengths = ion_df["wavelength"].values
    expansion_opacities = ion_df["expansion_opacity"].values
    grid = np.arange(lambda_min, lambda_max + lambda_bin, lambda_bin)
    expansion_opacity_histo, edges = np.histogram(
        wavelengths, bins=grid, weights=expansion_opacities
    )
    grid_midpoints = 0.5 * (grid[1:] + grid[:-1])
    return expansion_opacity_histo, grid_midpoints


def make_expansion_opacity_grid_energy(ion_df, e_min, e_max, n_bins=500):
    wavelengths = ion_df["wavelength"].values
    expansion_opacities = ion_df["expansion_opacity"].values
    grid_energy= np.linspace(e_min, e_max, n_bins)[::-1]
    grid=1e8/grid_energy
    expansion_opacity_histo, edges = np.histogram(
        wavelengths, bins=grid, weights=expansion_opacities
    )
    grid_midpoints = 0.5 * (grid[1:] + grid[:-1])
    return expansion_opacity_histo, grid_midpoints

##@nb.jit
def B_lambda(T, lambda_value):
    """
    Function to return the Planck energy density as function of wavelength, NOT frequency
    ----------
    Parameters:
    lambda_value: scalar, float
        wavelength
    """
    h_cgs=c.h.cgs
    c_cgs=c.c.cgs
    k_B_cgs=c.k_B.cgs
    lambda_cm = (lambda_value * u.Angstrom).to(u.cm)
    return (
        2
        * h_cgs
        * c_cgs ** 2
        / lambda_cm ** 5
        * 1
        / (np.exp(h_cgs * c_cgs / (k_B_cgs * T * lambda_cm)) - 1)
    )

def comp_Planck_opac(T, lambda_values, exp_opac_data):
    """
    Function to compute the Planck mean opacity at a specified temperature using 
	trapezoidal integration
    ----------
    Parameters:
    T: scalar, object (u.K)
        radiation temperature
    lambda_values: 1D array, float
        array of lambda values
	exp_opac_data: 1D array, float
		array of expansion opacity data for each lambda value
    """
    sigma = c.sigma_sb.to(u.erg / (u.s * u.K ** 4 * u.cm ** 2))
    Planck_opac_num = 0
    PlancK_opac_den = 0
    for idx, lambda_value in enumerate(lambda_values):
        if idx > 2:
            lambda_1 = (lambda_values[idx - 1] * u.Angstrom).to(u.cm).value
            lambda_2 = (lambda_values[idx] * u.Angstrom).to(u.cm).value
			# compute numerator with trapezoidal rule
            Planck_opac_num += (
                1
                / 2
                * (lambda_2 - lambda_1)
                * (
                    (
                        exp_opac_data[idx - 1]
                        * B_lambda(T, lambda_values[idx - 1])
                        / lambda_1 ** 2
                    )
                    + (
                        exp_opac_data[idx]
                        * B_lambda(T, lambda_values[idx])
                        / lambda_2 ** 2
                    )
                )
            )
			# compute denominator with trapezoidal rule
            PlancK_opac_den += (
                1
                / 2
                * (lambda_2 - lambda_1)
                * (
                    (B_lambda(T, lambda_values[idx - 1]) / lambda_1 ** 2)
                    + (B_lambda(T, lambda_values[idx]) / lambda_2 ** 2)
                )
            )
    return Planck_opac_num / PlancK_opac_den


if __name__ == "__main__":
    atomic_number = 60
    dir_path = "Database/FAC_data"
    filename = "test"  
    type_calc = "FAC"
    ion_stages = [1, 2]
    full_atomic_data, atomic_info = GetCompleteData(atomic_number, dir_path, filename, type_calc, ion_stages,)
    print(full_atomic_data)