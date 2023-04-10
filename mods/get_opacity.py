from tqdm import tqdm
import opacity_mod as op
import expansion_opacity as ExpansionOpac
import ion_balance as Ion_balance
import plankmeanopacity as PlankMeanOpacities

if __name__ == "__main__":
    atomic_number = 60
    dir_path = "Database/FAC_data"
    filename = "test"  
    type_calc = "FAC"
    ion_stages = [1, 2]

    extension_name = "test" # String identifying output files

    print ('Getting the data')
    # Read the data
    full_atomic_data, atomic_info = op.GetCompleteData(
        atomic_number, dir_path, filename, type_calc, ion_stages,)
    
    print ('Getting the ionization balance')
    # Get the ionization balance
    Ion_balance.Get_Ionic_balance(full_atomic_data, atomic_info, extension_name=extension_name)

    print ('Getting the opacity')
    # Getting opacity 
    for T in tqdm(range(4000, 6000, 1000)): ##Range of temperatures for which opacity is calculated
        ExpansionOpac.Get_Opacity(full_atomic_data, atomic_info, T=T, line_binned=False, lambda_bin=10, extension_name=extension_name)

    print ('Getting the plank mean opacities')
    PlankMeanOpacities.GetPlankOpacities(atomic_info, full_atomic_data,T_step=500, extension_name=extension_name)
    

