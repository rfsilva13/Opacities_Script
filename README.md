# How it works

Main file to compute opacities is `GetOpacity.py`. It works by reading the files (from FAC in this case) in the *Database* folder and computing the opacities. Output is saved in the *Opacities*, *Ionization_balance* and *Plank Opacities* folders, if all modules are ran (execution of any of those can be commented in the main file). Only computation of expansion opacity (the most commonly type used) is fully working at the moment (no line opacity computation is possible with this script, used for example, in *Fontes et al. 2020*)

# Naming conventions

**Levels files** should end with *.lev.asc*

**Transitions files** should end with *.tr.asc*  

At least the files for two consecutive ions should be included (most commonly doubly and triply ionized for the temperature range we are considering).

Example naming: NdII_<anything>.lev.asc; NdIII_<anything>.lev.asc; NdII_<anything>.tr.asc; NdIII_<anything>.tr.asc

Name starts with the name of the ion and the ionization stage, in roman numerals, folowed by a string which is used to distinguish files. That string can be anything but must be the same for all files which are used in the computation of the opacities. That string should then be supplied in the `filename` parameter when running the  `GetOpacity.py` module.

# Output files

## Ionization balance

First column is temperature, while remaning columns are the ion fraction in each of the 4 first ionization stages

## Expansion Opacity

First column is wavelength, second is opacity due to the ion in the first ionization stage (1+ / II), third column is opacity due to the ion in the second ionization stage (2+/ III).

## Plank opacity

First column is temperature, second is plank mean opacity due to the ion in the first ionization stage (1+ / II), third column is plank mean opacity due to the ion in the second ionization stage (2+/ III).

## Important note

The function ``download_ionization_energies`` from the carsus program has to be modified to avoid errors when parsing from nist. The new function is the following:

```python
def download_ionization_energies(
        spectra='h-uuh',
        e_out=0,
        e_unit=1,
        format_=1,
        at_num_out='on',
        sp_name_out='off',
        ion_charge_out='on',
        el_name_out='off',
        seq_out='off',
        shells_out='on',
        conf_out='off',
        level_out='on',
        ion_conf_out='off',
        unc_out='on',
        biblio='off'):
    """
        Downloader function for the Ionization Energies Data from the NIST Atomic Spectra Database
        Parameters
        ----------
        spectra: str
            (default value = 'h-uuh')
        Returns
        -------
        str
            Preformatted text data
        """
    data = {'spectra': spectra, 'units': e_unit,
            'format': format_, 'at_num_out': at_num_out, 'sp_name_out': sp_name_out,
            'ion_charge_out': ion_charge_out, 'el_name_out': el_name_out,
            'seq_out': seq_out, 'shells_out': shells_out, 'conf_out': conf_out,
            'level_out': level_out, 'ion_conf_out': ion_conf_out, 'e_out': e_out,
            'unc_out': unc_out, 'biblio': biblio}

    data = {k: v for k, v in data.items() if v is not 'off'}



    logger.info("Downloading ionization energies from the NIST Atomic Spectra Database.")
    r = requests.post(IONIZATION_ENERGIES_URL, data=data)
    print(r.url)
    return r.text
```
