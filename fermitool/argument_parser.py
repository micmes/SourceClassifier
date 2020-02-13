"""
This program is a simple interactive tool aimed at visualizing the Fermi 4FGL data. In theory, 
there are infinite filters, operations and plots that can be performed,
but we decided to limit the user's freedom to modify the data for the sake of simplicity.
There are four main categories of plots: Localization, Variability and Spectra.
The plots for each category are those implemented in the Fermi_Dataset class (see fermitool.py).
The plots will be saved in the output file.
The filtering operation consists in keeping the brightest sources of the catalog.
Enjoy!
"""


from argparse import ArgumentParser
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
    data_4FGL.dist_variability(savefig=True)
    data_4FGL.compare_variability(savefig=True)

#Spectral plots
if args.spectra:
    data_4FGL.dist_models(savefig=True)
    data_4FGL.plot_spectral_param(savefig=True)

# Localization plots
if args.localize:
    data_4FGL.galactic_map('galactic', title='LOC_all_sources', savefig=True, color='CLASS1')
    c = data_4FGL._df['CLASS1']
    #print(data_4FGL._df[['ASSOC_FGL','ASSOC_TEV','CLASS1','CLASS2','ASSOC1','ASSOC2']].isin(['pwn']).any())
    #print(data_4FGL.filtering((c == 'psr'))._df[['CLASS1','ASSOC1']])
    data_4FGL.filtering((c == 'psr') | (c == 'pwn')).galactic_map('galactic', title='LOC_psr', savefig=True, color='CLASS1')

print(args)