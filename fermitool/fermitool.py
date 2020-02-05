import os
from astropy.io import fits
from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
from matplotlib import pyplot as plt
import seaborn as sns
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
      This operation is useful for plots, when we don't need to distinguish between associated and identified sources.
      """
      self._df['CLASS1'] = self._df['CLASS1'].str.strip()
      self._df['CLASS1'] = self._df['CLASS1'].str.lower()
      
      return Fermi_Dataset(self._df)

  def source_hist(self, colname, title='Histogram', xlabel='x',
          ylabel='y', savefig=False, xlog=False, ylog=False, **kwargs):
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

    if xlog==True:
      plt.xscale('log')
    if ylog==True:
      plt.yscale('log')

    if savefig:
      # create dir if it doesn't exist
      if not os.path.isdir(output_path):
        os.makedirs(output_path)
      plt.savefig(output_path + '/' + title + '.png')
    else:
      plt.show()
    return

  def galactic_map(self, coord_type='equatorial', title='Galactic Map', savefig=False, c=None,
           colorbar=False, **kwargs):
    """
    Plot a galactic map given sources. We're assuming that the right
    ascension and the declination columns are labeled by 'RAJ2000' and
    'DEJ2000' respectively.
    :coord_type: type of the given coordinates. String values are admitted:
    'equatorial' (default) or 'galactic'.
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

    if coord_type == 'equatorial':
        lon = d['RAJ2000']
        lat = d['DEJ2000']
    if coord_type == 'galactic':
        lon = d['GLON']
        lat = d['GLAT']
    else:
        print('not valid') # to be corrected: raise an error

    lon = coord.Angle(lon * u.degree)
    lon = lon.wrap_at(180 * u.degree)
    lat = coord.Angle(lat * u.degree)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    ax1 = ax.scatter(lon.radian, lat.radian, c=c, **kwargs)

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

  def dist_models(self, title='Distribution of Spectral Models', savefig=False, **kwargs):
    """
    Plots the bar chart of the 3 models of the sources.
    """
    y_pos = np.arange(3)
    objects= ('PowerLaw', 'LogParabola', 'PLSuperExpCutoff')
    occurences = self.df['SpectrumType'].value_counts()
    
    fig = plt.figure()
    plt.bar(y_pos, occurences, align='center', **kwargs)
    plt.xticks(y_pos, objects)
    plt.title(title)
    plt.ylabel('Number of sources')
    
    if savefig:
      # create dir if it doesn't exist
      if not os.path.isdir(output_path):
        os.makedirs(output_path)
      fig.savefig(output_path + '/' + title + '.png')
    else:
      fig.show()
    return
    
    
  def plot_spectral_param(self, title='Spectral Parameters', savefig=False, **kwargs):
      """
      Plot the spectral parameters of the sources.
      :title: title of the plot
      :savefig: choose whether to save the fig or not (in the output folder)
      :kwargs: set the points parameters according to 'matplotlib.pyplot.scatter' module
      """
      self.clean_classes()
      
      data1 = self._df[['PLEC_Index', 'PLEC_Expfactor', 'CLASS1']]
      data2 = self._df[['LP_Index', 'LP_beta', 'CLASS1']]
      
      fig, (ax1, ax2) = plt.subplots(1, 2)
      sns.scatterplot(x='PLEC_Index', y='PLEC_Expfactor', hue='CLASS1', data=data1, ax=ax1)
      sns.scatterplot(x='LP_Index', y='LP_beta', hue='CLASS1', data=data2, ax=ax2)

      ax1.legend().set_visible(False)
      ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
      plt.tight_layout()

      if savefig:
      # create dir if it doesn't exist
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        fig.savefig(output_path + '/' + title + '.png')
      else:
        fig.show()
      return

  def energyflux_map(self, title='Energy Flux map', savefig=False, **kwargs):
    """
    Plot the galactic map with the energy flux between 100 MeV and 100 GeV as gradient.
    """
    self.galactic_map(title=title, savefig=savefig, c='Energy_Flux100', colorbar=True)
    return
  
  def dist_variability(self, title='Distribution of the variability index', savefig=False, **kwargs):
    """
    Plot the distribution of the variability index for the sources.
    """
    self.source_hist(colname='Variability_Index', title=title, xlabel='Variability Index',
                      ylabel='Number of sources', savefig=savefig, xlog=True, ylog=True,
                      range=(0,10000), bins=100, histtype='step')
    
  def compare_variability(self, title='Comparison of Varibility index for 2 month intervals and that for 12 months',
                          savefig=False, **kwargs):
    """
    Plot Variability Index 2 month vs Variability Index 12 month.
    """
    x = self._df['Variability_Index']
    y = self._df['Variability2_Index']

    fig = plt.figure()
    plt.scatter(x, y, marker='+')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('12 month')
    plt.ylabel('2 month')

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
  print(data_4FGL.df['GLON'])
  print(data_4FGL.df['GLAT'])

  # prove
  #data_4FGL.filtering(data_4FGL.df['CLASS1'].str.match('(psr)|(PSR)'))
    