import pathlib
import warnings
import astropy.units as u
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import mods.opacity_mod as op

except:
    import opacity_mod as op

warnings.filterwarnings("ignore")

# @profile
def GetPlankOpacities(
    atomic_info,
    full_atomic_data,
    T_min=1000,
    T_max=20000,
    T_step=500,
    rho=1e-13,
    time=1,
    lambda_bin=500,
    extension_name="",
):
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
    rho = rho * u.g / u.cm ** 3
    time = time * u.day.to(u.s)
    save_dir = pathlib.Path("PlankOpacities")
    save_dir.mkdir(exist_ok=True)
    columns = ["T"] + [i for i in range(1, len(ion_stages) + 1)]
    plank_opacitiesdf = pd.DataFrame(columns=columns)
    Ts = np.arange(T_min, T_max, T_step)
    plank_opacitiesdf["T"] = Ts
    plank_opacitiesdf.set_index("T", inplace=True)
    for T in tqdm(range(T_min, T_max, T_step)):
        T = T * u.K
        exp_op = op.compute_expansion_opacity(
            atomic_number,
            ion_stages,
            lambda_bin,
            0,
            time,
            T,
            rho,
            ground_levels,
            gfall_levels,
            levels,
            atomic_weights,
            ionization_energies,
            lines,
            line_binned=False,
        )
        for i in ion_stages:
            opacity, grid_midpoints = op.make_expansion_opacity_grid(
                exp_op.loc[atomic_number, i, :], 0, 25000, lambda_bin
            )
            Plankopacity = op.comp_Planck_opac(T, grid_midpoints, opacity)
            plank_opacitiesdf[i][T.value] = Plankopacity
    plank_opacitiesdf.to_csv(
        str(save_dir) +"/"+ "PlankOpacities_"
        + str(atomic_number)
        + "_"
        + str(type_calc)
        + "_"+extension_name+".csv",
        index=True,
        )
    return plank_opacitiesdf


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
    GetPlankOpacities(atomic_info, full_atomic_data,T_step=1000)
    print("Done")
