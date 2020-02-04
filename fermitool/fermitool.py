import os
from astropy.io import fits
from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
from matplotlib import pyplot as plt
import numpy as np

# requires setup.sh to run
source_root = os.environ["SOURCE_ROOT"]
output_path = source_root + '/output'


class Fermi_Dataset:
  """
  Class capable to perform data analysis and to visualize the given dataset in a graphical fashion.
  The Fermi_Dataset class is targeted at the 4FGL forth source catalog.
  """

  def __init__(self, data):
    """
    Constructor.
    :data: pandas dataframe containing the data
    """
    self._df = data

  @property
  def df(self):
    return self._df

  def filtering(self, df_condition):
    """
    Selects dataframe rows based on df_condition.
    """
    return Fermi_Dataset(self._df[df_condition])

  def col(self, colname):
    """
    Return the content of a given column.
    """
    return self._df[colname]

  def columns(self):
    """
    Return column names.
    """
    return self._df.columns
  
  def clean_classes(self):
      """
      Removes extra spaces and lowers all the characters in the CLASS1 column of the dataframe.
      """
      self._df['CLASS1'] = self._df['CLASS1'].str.strip()
      self._df['CLASS1'] = self._df['CLASS1'].str.lower()
      
      return 

  def source_hist(self, colname, title='Histogram', xlabel='x',
          ylabel='y', savefig=False, **kwargs):
    """
    This method provides a histogram plot given a single array in
    input. Most of the features are inherited from the matplotlib hist
    function.
    :colname: The name of the column to plot (str)
    :filter: Boolean value. If true, plot data from filtered_df
    :title: the title of the histogram shown in the plot (str)
    :xlabel: x label shown in the plot (str)
    :ylabel: y label shown in the plot (str)
    :savefig: choose whether to save the fig or not
    :kwargs: the same parameters of the plt.hist function (str)
    """
    plt.figure()
    plt.hist(self.df[colname], **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if savefig:
      # create dir if it doesn't exist
      if not os.path.isdir(output_path):
        os.makedirs(output_path)
      plt.savefig(output_path + '/' + title + '.png')
    else:
      plt.show()
    return

  def galactic_map(self, title='Galactic Map', savefig=False, c=None,
           colorbar=False, **kwargs):
    """
    Plot a galactic map given sources. We're assuming that the right
    ascension and the declination columns are labeled by 'RAJ2000' and
    'DEJ2000' respectively.
    :title:the title of the histogram shown in the plot (str)
    :savefig: choose whether to save the fig or not (in the output folder)
    :c: colours in the scatter plot. Every value that satisfies the
    'matplot.pyplot.scatter' conditions is accepted; in addition,
    string values that match the column names give a gradient according
    to values in the column itself.
    :kwargs: set the points parameters according to 'matplotlib.pyplot.scatter' module
    """

    d = self._df

    # check whether 'c' is a column name. If yes, use the values in
    # that column to color the scatter plot
    if c in self.columns():
      c = d[c]

    ra = coord.Angle(d['RAJ2000'] * u.degree)
    ra = ra.wrap_at(180 * u.degree)
    dec = coord.Angle(d['DEJ2000'] * u.degree)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    ax1 = ax.scatter(ra.radian, dec.radian, c=c, **kwargs)

    if colorbar:
      fig.colorbar(ax1)
    plt.title(title)

    if savefig:
      # create dir if it doesn't exist
      if not os.path.isdir(output_path):
        os.makedirs(output_path)
      fig.savefig(output_path + '/' + title + '.png')
    else:
      fig.show()
    return

  def dist_models(self):
    """
    Plots the distribution of the 3 models of the sources.
    """
    self.source_hist('SpectrumType', title='Distribution of Models',
             xlabel='Spectrum Model', ylabel='Number of sources',
             bins=3, histtype='bar')
    
  def plot_spectral_param(self, title='Spectral Parameters', savefig=False, **kwargs):
      """
      Plot the spectral parameters of the sources.
      :title: title of the plot
      :savefig: choose whether to save the fig or not (in the output folder)
      :kwargs: set the points parameters according to 'matplotlib.pyplot.scatter' module
      """
      self.clean_classes
      
      x1 = self.df['PLEC_Index']
      y1 = self.df['PLEC_Expfactor']
      x2 = self.df['LP_Index']
      y2 = self.df['LP_beta']
      
      fig, (ax1, ax2) = plt.subplots(1, 2)
      
      ax1.scatter(x1, y1)
      ax1.set_yscale('log')
      
      ax2.scatter(x2, y2)
      ax2.set_yscale('log')
      
      if savefig:
      # create dir if it doesn't exist
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        fig.savefig(output_path + '/' + title + '.png')
      else:
        fig.show()
      return

if __name__ == '__main__':
  # import fits file
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
  print(data_4FGL.df.columns)
  #E_peak = data_4FGL.df['Pivot_Energy'] * np.exp( (2-data_4FGL.df['LP_Index']) / (2* data_4FGL.df['LP_beta']))
  
  
  
  brightest_sources = data_4FGL.filtering(data_4FGL.df['Signif_Avg']>=30)
  
  #brightest_sources.plot_spectral_param(savefig=True)
  
  
  # prove
  #data_4FGL.filtering(data_4FGL.df['CLASS1'].str.match('(psr)|(PSR)'))
    