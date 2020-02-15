import os

from astropy.io import fits
from astropy.table import Table

from fermitool import Fermi_Dataset

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

  # print some features
  '''
  print(data_4FGL.df.columns)
  print(data_4FGL.df)
  print(data_4FGL.df['GLAT'])
  print(extended_4FGL.df.columns)
  '''

  #data_4FGL.classifier()
  #data_4FGL.source_hist(['Conf_95_SemiMajor','Conf_95_SemiMinor'], savefig=True, title='LOCH_error_radii', bins=50, range=(0,0.2))
  #print(data_4FGL.filtering(data_4FGL.df['CLASS1'] == 'unassociated').df['CLASS1'])

  print(data_4FGL.df.values.T[0])
'''
  # prove
  # data_4FGL.filtering(data_4FGL.df['CLASS1'].str.match('(psr)|(PSR)'))

  # LOCALIZATION GRAPHS
  # LOCM stands for galactic map localization plots; LOCH is referred to
  # histograms
  data_4FGL.galactic_map(coord_type='galactic', title='LOCM_all_sources', savefig=True,
               color='CLASS1', marker='.', s=50)
  c = data_4FGL._df['CLASS1']
  data_4FGL.filtering((c == 'psr') | (c == 'pwn')).galactic_map('galactic', title='LOCM_psr_pwn', savefig=True,
                                                                color='CLASS1', marker='.', s=50)
  data_4FGL.filtering(c == 'psr').source_hist('GLAT', savefig=True, title='LOCH_GLAT_psr', xlabel='GLAT', ylabel='Counts',
                                              bins=40, range=(-90,90))
  data_4FGL.filtering(c == 'pwn').source_hist('GLAT', savefig=True, title='LOCH_GLAT_pwn', xlabel='GLAT',
                                              ylabel='Counts', bins=40, range=(-90,90))
  # notice that here we're investigating another fits extension
  extended_4FGL.galactic_map('galactic', title='LOCM_extension', savefig=True, color='Model_SemiMajor', marker='.', s=70)
'''
