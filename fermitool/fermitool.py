import os
from astropy.io import fits
from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from pandas.api.types import is_numeric_dtype
import pandas as pd
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


    :param data:  pandas dataframe containing the data
    """
    self._df = data

  @property
  def df(self):
    return self._df
  
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

  def filtering(self, df_condition):
    """
    Selects dataframe rows based on df_condition.

    :param df_condition: The condition based on which you filter the dataframe. 
    """
    return Fermi_Dataset(self._df[df_condition])
  
  def clean_classes(self):
    """
    Removes extra spaces and lowers all the characters in the CLASS1 column of the dataframe.
    This operation is useful for plots, when we don't need to distinguish between associated and identified sources.
    """
    self._df = self._df.assign(CLASS1=self._df['CLASS1'].apply(lambda x: x.strip().lower()))
    return Fermi_Dataset(self._df)

  def clean_nan(self, col):
    #self._df = self._df.dropna(subset=[col])
    self._df = self._df.fillna(self._df.mean())
    return Fermi_Dataset(self._df)
    
  def show_plot(self, savefig=False, title='Title'):
    """
    Shows the plot or saves the figure in the output folder. This method is always called
    when we make a plot.


    :param savefig: if True, save the figure in the output directory
    :param title: title of the plot
    """
    if savefig:
      # create dir if it doesn't exist
      if not os.path.isdir(output_path):
        os.makedirs(output_path)
      plt.savefig(output_path + '/' + title + '.png')
    else:
      plt.show()

  def source_hist(self, colname, title='Histogram', xlabel='x',
          ylabel='y', savefig=False, xlog=False, ylog=False, **kwargs):
    """
    This method provides a histogram plot given a single array in
    input. Most of the features are inherited from the matplotlib hist
    function.


    :param colname:  The name of the column to plot (str)
    :param filter:  Boolean value. If true, plot data from filtered_df
    :param title:  the title of the histogram shown in the plot (str)
    :param xlabel:  x label shown in the plot (str)
    :param ylabel:  y label shown in the plot (str)
    :param savefig:  choose whether to save the fig or not
    :param xlog: if True, set xscale to log
    :param ylog: if True, set yscale to log
    :param kwargs:  the same parameters of the plt.hist function (str)
    """
    plt.figure()
    plt.hist(self.df[colname], **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xlog:
      plt.xscale('log')
    if ylog:
      plt.yscale('log')

    self.show_plot(savefig=savefig, title=title)

  def dist_models(self, title='Distribution of Spectral Models', savefig=False, **kwargs):
    """
    Plots the bar chart of the 3 models of the sources.


    :param title: title of the plot
    :param savefig: if True, save figure in output directory
    :param kwargs: kwargs of matplotlib.pyplot.bar
    """
    y_pos = np.arange(3)
    objects= ('PowerLaw', 'LogParabola', 'PLSuperExpCutoff')
    occurences = self.df['SpectrumType'].value_counts()
    
    fig = plt.figure()
    plt.bar(y_pos, occurences, align='center', **kwargs)
    plt.xticks(y_pos, objects)
    plt.title(title)
    plt.ylabel('Number of sources')
    
    self.show_plot(savefig=savefig, title=title)
    
  def plot_spectral_param(self, title='Spectral Parameters', savefig=False, **kwargs):
    """
    Plot the spectral parameters of the sources.


    :param title:  title of the plot
    :param savefig:  choose whether to save the fig or not (in the output folder)
    :param kwargs:  set the points parameters according to 'matplotlib.pyplot.scatter' module
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

    self.show_plot(savefig=savefig, title=title)

  def energyflux_map(self, title='Energy Flux map', savefig=False, **kwargs):
    """
    Plot the galactic map with the energy flux between 100 MeV and 100 GeV as gradient.


    :param title: title of the plot
    :param savefig: if True, save figure in the output directory
    """
    self.galactic_map(title=title, savefig=savefig, c='Energy_Flux100', colorbar=True)
    return
  
  def dist_variability(self, title='Distribution of the variability index', savefig=False, **kwargs):
    """
    Plot the distribution of the variability index for the sources.


    :param title: title of the plot
    :param savefig: if True, save figure in the output directory
    :param kwargs: kwargs of matplotlib.pyplot.hist module
    """
    self.clean_nan('Variability_Index')
    
    self.source_hist(colname='Variability_Index', title=title, xlabel='Variability Index',
                      ylabel='Number of sources', savefig=savefig, xlog=True, ylog=True,
                      range=(0,500), bins=200, histtype='step')
    
  def compare_variability(self, title='Comparison of Varibility index for 2 month intervals and that for 12 months',
                          savefig=False, **kwargs):
    """
    Plot Variability Index 2 month vs Variability Index 12 month.


    :param title: title of the plot
    :param savefig: if True, save figure in the output directory
    :param kwargs: kwargs of matplotlib.pyplot.scatter module
    """
    x = self._df['Variability_Index']
    y = self._df['Variability2_Index']

    fig = plt.figure()
    plt.scatter(x, y, marker='+', **kwargs)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('12 month')
    plt.ylabel('2 month')

    self.show_plot(savefig=savefig, title=title)

  def galactic_map(self, coord_type='equatorial', title='Galactic Map',
           savefig=False, color=None, palette=None, marker='None',
           alpha=None):
    """
    Plot a galactic map given sources. We're assuming that the right
    ascension and the declination columns are labeled by 'RAJ2000' and
    'DEJ2000' respectively.

    
    :param coord_type:  type of the given coordinates. String values are admitted:
                  'equatorial' (default) or 'galactic'.
    :param title:  the title of the histogram shown in the plot (str)
    :param savefig:  choose whether to save the fig or not (in the output folder)
    :param color:  the name of the column to color the points. It can be a
             numeric value or a string value.

    """

    self.clean_classes()

    # choose which type of coordinates i want to plot
    color_label = color
    if coord_type == 'equatorial':
      lon_label = 'RAJ2000'
      lat_label = 'DEJ2000'
    if coord_type == 'galactic':
      lon_label = 'GLON'
      lat_label = 'GLAT'
    else:
      print('not valid')  # to be corrected: raise an error
    lon = self._df[lon_label]
    lat = self._df[lat_label]
    col = self._df[color_label]

    # convert deg values to RA
    lon = coord.Angle(lon * u.degree)
    lon = lon.wrap_at(180 * u.degree).radian
    lat = coord.Angle(lat * u.degree).radian


    # build dataframe
    coord_df = pd.DataFrame({lon_label:lon, lat_label:lat, color_label:col})

    fig, ax = plt.subplots(1, 1)
    ax = plt.axes(projection='mollweide')
    ax.grid(b=True)

    #if values are discrete, than plot a legend
    if color == 'CLASS1':
      sns.scatterplot(x=lon_label, y=lat_label, hue=color_label,
              data=coord_df, ax=ax, palette=palette,
              markers=marker, alpha=alpha)
      ax.set_position(pos = [0.15, 0.2, 0.6, 0.6])
      ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), prop={'size':9},
            fancybox=True, shadow=True)
    # else plot a colorbar
    elif color in self.columns() and is_numeric_dtype(self._df[color_label]):
      scat = ax.scatter(lon, lat, c=col.tolist(), cmap=palette, marker=marker,
                alpha=alpha)
      ax.set_xlabel(lon_label)
      ax.set_ylabel(lat_label)
      cbar = fig.colorbar(scat)
      cbar.set_label(color_label)
    # else draw with no colors
    else:
      if color is not None:
        print('Warning: not valid value for color column') #raise warning
      ax.scatter(lon, lat, marker=marker, alpha=alpha)

    ax.set_title(title)


    self.show_plot(savefig=savefig, title=title)


if __name__ == '__main__':
  # import fits file
  data_path = os.environ["SOURCE_ROOT"] + '/data/gll_psc_v21.fit'

  try:  # does the file exist? If yes
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
  # data_4FGL.filtering(data_4FGL.df['CLASS1'].str.match('(psr)|(PSR)'))

  # LOCALIZATION GRAPHS
  # all sources:
  data_4FGL.galactic_map(coord_type='galactic', title='All_sources',
               color='CLASS1', savefig=True)