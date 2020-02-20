"""
This program is a simple interactive tool aimed at visualizing the Fermi 4FGL data. In theory, 
there are infinite filters, operations and plots that can be performed,
but we decided to limit the user's freedom to modify the data for the sake of simplicity.
There are three main categories of plots: Localization, Variability and Spectra.
The plots for each category are those implemented in the Fermi_Dataset class (see fermitool.py).
The plots will be saved in the output file.
The filtering operation consists in keeping the brightest sources of the catalog.
Enjoy!
"""


from argparse import ArgumentParser
from astropy.io import fits
from astropy.table import Table
from fermitool import *

data_path = os.environ["SOURCE_ROOT"] + '/data/gll_psc_v21.fit'

try: # does the file exist? If yes
    with fits.open(data_path) as hdul:
      fits_data = hdul[1].data
except OSError as e:
    print(e)

t_astropy = Table(fits_data)

col1D = [col1D for col1D in t_astropy.colnames if len(t_astropy[col1D].shape) <= 1]
data = t_astropy[col1D].to_pandas()

# Set some plot kwargs
map_kwargs = {"cmap" : 'hsv',
            "savefig" : True,
            "marker" : '.',
            "s" : 50}

hist_kwargs = {"savefig" : True,
                "bins" : 40,
                "histtype" : 'step'}


# define an istance
data_4FGL = Fermi_Dataset(data)

#Make parser and add arguments
parser = ArgumentParser(description=__doc__)

parser.add_argument("--columns", action="store_true", 
                    help="show all the column names")
parser.add_argument("--brightest", action="store_true", 
                    help="keep only the brightest sources in the dataset")
parser.add_argument("--variability", action="store_true", 
                    help="plot the distribution of the variability index and the comparison of the 2month and 12 month variability indexes")
parser.add_argument("--spectra", action="store_true",
                    help= "plot the distribution of the 3 models and the spectral parameters")
parser.add_argument('--localize', action='store_true',
                    help='plot the position of the sources, divided by type')
args = parser.parse_args()

#Show columns
if args.columns:
    print(data_4FGL.df.columns)

#Filter on bright sources
if args.brightest:
    data_4FGL = data_4FGL.filtering(data_4FGL.df['Signif_Avg']>=30)

#Variability plots
if args.variability:
    data_nan_removed = data_4FGL.remove_nan_rows(['Variability_Index'])
    data_nan_removed.source_hist(colname='Variability_Index', title='VAR_distribution_of_variability', savefig=True,
                               xlog=True, ylog=True, range=(0,500), bins=200, histtype='step')
    data_nan_removed.compare_variability(title='VAR_Comparison of Varibility index for 2 month intervals and that for 12 months', savefig=True)

#Spectral plots
if args.spectra:
    data_4FGL.filtering(data_4FGL.df['Signif_Avg']>30).plot_spectral_param(title='SPEC_Spectral Parameters',savefig=True)
    data_4FGL.dist_models(title='SPEC_Distribution of the spectral models', savefig=True)

# Localization plots
if args.localize:
    data_4FGL.galactic_map(coord_type='galactic', title='LOCM_all_sources',
                            color='CLASS1', **map_kwargs)
    data_4FGL_cleaned = data_4FGL.clean_column('CLASS1')
    data_4FGL_psr_pwn = data_4FGL_cleaned.filtering(data_4FGL_cleaned.df['CLASS1'].str.match('(psr)|(pwn)'))

    data_4FGL_psr_pwn.galactic_map('galactic', title='LOCM_psr_pwn',
                                    color='CLASS1', **map_kwargs)
