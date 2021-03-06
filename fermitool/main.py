import os
from functools import reduce

from astropy.io import fits
from astropy.table import Table

from fermitool import Fermi_Dataset

import numpy as np

if __name__ == '__main__':

  # import fits file
  data_path = os.environ["SOURCE_ROOT"] + '/data/gll_psc_v21.fit'

  try:  # does the file exist? If yes
    with fits.open(data_path) as hdul:
      main_fits = hdul[1].data
      extended_source_fits = hdul[2].data
  except OSError as e:
    print(e)

  t_astropy = Table(main_fits)
  te_astropy = Table(extended_source_fits)

  col1D = [col1D for col1D in t_astropy.colnames if len(t_astropy[col1D].shape) <= 1]
  data = t_astropy[col1D].to_pandas()
  data_4FGL = Fermi_Dataset(data)

  col1D = [col1D for col1D in te_astropy.colnames if len(te_astropy[col1D].shape) <= 1]
  data = te_astropy[col1D].to_pandas()
  extended_4FGL = Fermi_Dataset(data)

  # Set some plot kwargs
  map_kwargs = {"savefig" : True,
                "marker" : '.',
                "s" : 50}

  hist_kwargs = {"savefig" : True,
                 "bins" : 40,
                 "histtype" : 'step'}

  # VARIABILITY
  data_nan_removed = data_4FGL.remove_nan_rows(['Variability_Index'])
  data_nan_removed.source_hist(colname='Variability_Index', title='VAR_distribution_of_variability', savefig=True,
                               xlog=True, ylog=True, range=(0,500), bins=200, histtype='step')
  data_nan_removed.compare_variability(title='VAR_Comparison of Varibility index for 2 month intervals and that for 12 months', savefig=True)
  
  # LOCALIZATION GRAPHS
  # LOCM stands for galactic map localization plots; LOCH is referred to
  # histograms
  data_4FGL.galactic_map(coord_type='galactic', title='LOCM_all_sources',
                         color='CLASS1', **map_kwargs)
  data_4FGL_cleaned = data_4FGL.clean_column('CLASS1')
  data_4FGL_psr_pwn = data_4FGL_cleaned.filtering(data_4FGL_cleaned.df['CLASS1'].str.match('(psr)|(pwn)'))

  data_4FGL_psr_pwn.galactic_map('galactic', title='LOCM_psr_pwn',
                                 color='CLASS1', **map_kwargs)

  data_4FGL_cleaned.filtering(data_4FGL_cleaned.df['CLASS1'] == 'psr').source_hist('GLAT', title='LOCH_GLAT_psr',
                                               range=(-90,90), **hist_kwargs)
  data_4FGL_cleaned.filtering(data_4FGL_cleaned.df['CLASS1'] == 'pwn').source_hist('GLAT', title='LOCH_GLAT_pwn',
                                               range=(-90,90), **hist_kwargs)
  # notice that here we're investigating another fits extension
  extended_4FGL.galactic_map('galactic', title='LOCM_extension', color='Model_SemiMajor', **map_kwargs)

  # define geometric mean
  def geometric_mean(*args):
    # takes n columns as inputs
    n = len(args)
    return np.power(np.array([x * y for x, y in zip(*args)]), 1./n)

  # define new column 'geometric_mean' and then plot source_hist
  df_remove_nan = data_4FGL.remove_nan_rows(['Conf_95_SemiMajor','Conf_95_SemiMinor'])
  df_geom = df_remove_nan.def_column(['Conf_95_SemiMajor','Conf_95_SemiMinor'], geometric_mean, 'Geom_mean')
  df_geom.source_hist('Geom_mean', title='LOCH_error_radii', **hist_kwargs)

  #SPECTRAL PLOTS
  data_4FGL_cleaned.filtering(data_4FGL_cleaned.df['Signif_Avg']>30).plot_spectral_param(title='SPEC_Spectral Parameters',savefig=True)
  data_4FGL.dist_models(title='SPEC_Distribution of the spectral models', savefig=True)

  # CLASSIFICATION
  data_4FGL.classifier(True)

  