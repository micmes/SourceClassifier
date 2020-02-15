import os

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import astropy.coordinates as coord
import astropy.units as u
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn import metrics
from collections import Counter

import warnings
import logging
logging.basicConfig(level=logging.INFO)

# requires setup.sh to run
source_root = os.environ["SOURCE_ROOT"]
output_path = source_root + '/output'


def show_plot(savefig=False, title='Title'):
  """
  Shows the plot or saves the figure in the output folder. This function is always called
  when we make a plot. 

  :param savefig: if True, save the figure in the output directory
  :param title: title of the plot
  """
  if savefig:
    # create dir if it doesn't exist
    if not os.path.isdir(output_path):
      os.makedirs(output_path)
    plt.savefig(output_path + '/' + title + '.png')
    logging.info('Image saved in SourceClassifier/output.')
  else:
    plt.show()

class Fermi_Dataset:
  """
  Class capable to perform data analysis and to visualize the given dataset in a graphical fashion.
  The constructer takes a DataFrame as an argument. 
  The Fermi_Dataset class is targeted at the 4FGL forth source catalog: all the column names are 
  those of the 4FGL. We therefore suggest to take a look at the following link <https://arxiv.org/abs/1902.10045>
  in order to get familiar with the data.
  """

  def __init__(self, data):
    """
    Constructor.

    :param data: pandas dataframe containing the data
    """
    self._df = data

  def _isnum(self, colname):
    return self.df[colname].is_numeric_dtype()

  def _isstr(self, colname):
    return self.df[colname].is_string_dtype()

  def _validcolumn(self, colname):
    if colname in self.df.columns:
      return True
    else:
      return False

  def def_column(self, colnames, func, newcolname='TEMP'):
    """
    This method provides a tool to create a new column applying a function
    that takes two or more columns as parameters. This is useful for example
    in evaluating geometric mean in order to draw the error radii histogram.
    """
    new_df = self.df
    new_df[newcolname] = self.df.apply(lambda row: func(row[colnames]))
    return Fermi_Dataset(new_df)

  @property
  def df(self):
    """
    Property prevent the user to accidentally modify the instance.
    """
    return self._df


  def filtering(self, df_condition, clean=True):
    """
    Selects dataframe rows based on df_condition. Returns an object with the filtered data.

    :param df_condition: The condition based on which you filter the dataframe.
    Make sure that the condition is written conformly to the pandas module, for example:
    Fermi_Dataset_Instance.df['Column_Name'] ><= value
    :param clean: choose whether to clean column or not. With cleaning, we mean
    - remove blank spaces
    - set all the uppercase to lowercase
    - set all the empty values with

    """
    try:
      if clean:
        df = self.clean_column('CLASS1').df
      else:
        df = self.df
      return Fermi_Dataset(df[df_condition])
    except Exception as e:
      print('Oops! Give me a valid condition', e)
      raise

  def clean_column(self, colname):
    """
    Removes extra spaces and lowers all the characters in the CLASS1 column of the dataframe.
    The empty rows are replaced with _unassociated_.
    This operation is useful for plots, when we don't need to distinguish between associated and identified sources(in CAPS).
    """
    if is_string_dtype(self._df[colname]):
      self._df = self._df.assign(CLASS1=self._df['CLASS1'].apply(lambda x: x.strip().lower()))
      self._df['CLASS1'] = self._df['CLASS1'].replace('', 'unassociated')
      logging.info('Column cleaned successfully.')
    else:
      logging.info('Column is numeric type: no cleaning required. Proceeding')
    return Fermi_Dataset(self._df)



  def remove_nan_rows(self, col):
    """
    Given the column name, remove the rows of the dataframe where the values of that column are NaN.

    :param col: name of the column to clean
    """
    try:
      self._df = self._df.dropna(subset=[col])
      logging.info('Cleaned NaN rows.')
      return Fermi_Dataset(self._df)
    except KeyError as e:
      print('Oops! Seems like you got the column name {} wrong. To see column names, type print(Obj.df.columns)'.format(e))
      raise

  def source_hist(self, colname, title='Histogram', xlabel='x',
          ylabel='y', savefig=False, xlog=False, ylog=False, **kwargs):
    """
    This method provides a histogram plot given a single array in
    input. Most of the features are inherited from the matplotlib hist
    function.

    :param colname:  The name of the column to plot. The column must be
    numeric.
    :param title:  the title of the histogram shown in the plot
    :param xlabel:  x label shown in the plot
    :param ylabel:  y label shown in the plot
    :param savefig:  choose whether to save the fig or not
    :param xlog: if True, set xscale to log
    :param ylog: if True, set yscale to log
    :param kwargs:  other parameters are passed to matplotlib.pyplot.hist
    module
    """
    assert colname in self.df.columns, 'Column name does not match any column ' \
                                       'in the dataframe. To see column names, ' \
                                       'try print(Obj.df.columns).'
    assert is_numeric_dtype(self.df[colname]), 'Column must be numeric type'

    data = self.df[colname]

    logging.info('Preparing the histogram...')
    plt.figure()
    plt.hist(data, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xlog:
      plt.xscale('log')
    if ylog:
      plt.yscale('log')

    logging.info('Histogram ready to be shown or saved!')
    show_plot(savefig=savefig, title=title)


  def dist_models(self, title='Distribution of Spectral Models', savefig=False, **kwargs):
    """
    Plots the bar chart of the 3 models of the sources.

    :param title: title of the plot
    :param savefig: if True, save figure in output directory
    :param kwargs: kwargs of matplotlib.pyplot.bar
    """
    y_pos = np.arange(3)
    objects= ('PowerLaw', 'LogParabola', 'PLSuperExpCutoff')
    logging.info('Counting the spectral models...')
    occurences = self.df['SpectrumType'].value_counts()

    logging.info('Preparing the chart plot...')
    plt.figure()
    plt.bar(y_pos, occurences, align='center', **kwargs)
    plt.xticks(y_pos, objects)
    plt.title(title)
    plt.ylabel('Number of sources')

    logging.info('Distribution of models ready to be shown or saved!')
    show_plot(savefig=savefig, title=title)


  def plot_spectral_param(self, title='Spectral Parameters', savefig=False, **kwargs):
    """
    Plot the spectral parameters of the sources.

    :param title:  title of the plot
    :param savefig:  choose whether to save the fig or not (in the output folder)
    :param kwargs:  set the points parameters according to 'matplotlib.pyplot.scatter' module
    """
    self.clean_column('CLASS1')

    logging.info('Preparing data for the plot...')
    data1 = self._df[['PLEC_Index', 'PLEC_Expfactor', 'CLASS1']]
    data2 = self._df[['LP_Index', 'LP_beta', 'CLASS1']]

    logging.info('Preparing the plot...')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.scatterplot(x='PLEC_Index', y='PLEC_Expfactor', hue='CLASS1', data=data1, ax=ax1)
    sns.scatterplot(x='LP_Index', y='LP_beta', hue='CLASS1', data=data2, ax=ax2)

    ax1.legend().set_visible(False)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()

    logging.info('Spectral parameters plot ready to be shown or saved!')
    show_plot(savefig=savefig, title=title)


  def dist_variability(self, title='Distribution of the variability index', savefig=False, **kwargs):
    """
    Plot the distribution of the variability index for the sources.

    :param title: title of the plot
    :param savefig: if True, save figure in the output directory
    :param kwargs: kwargs of matplotlib.pyplot.hist module
    """
    self.remove_nan_rows('Variability_Index')

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
    logging.info('Preparing data for variability plot...')
    x = self._df['Variability_Index']
    y = self._df['Variability2_Index']

    logging.info('Preparing the variability plot...')
    plt.figure()
    plt.scatter(x, y, marker='+', **kwargs)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('12 month')
    plt.ylabel('2 month')

    logging.info('Variability plot ready to be shown or saved!')
    show_plot(savefig=savefig, title=title)

  def galactic_map(self, coord_type='equatorial', title='Galactic_Map',
           savefig=False, color=None, **kwargs):
    """
    Plot a galactic map given sources. We're assuming that the right
    ascension and the declination columns are labeled by 'RAJ2000' and
    'DEJ2000' respectively, or 'GLAT' and 'GLON' in case galactic
    coordinates are requested.


    :param coord_type:  type of the given coordinates. String values are admitted:
                        'equatorial' (default) or 'galactic'.

    :param title: the title of the histogram shown in the plot (string type)
    :param savefig: choose whether to save the fig or not (in the output directory)
    :param color: the name of the column to color the points. If string
                  value, the axes will be drawn with 'seaborn.scatterplot' module,
                  while if the column is numeric type 'matplotlib.pyplot.scatter' will
                  take its place. (NOT totally working yet!)
    :param **kwargs: other parameters passed directly to
                    'seaborn.scatterplot' or 'matplotlib.pyplot.scatter' depending on the
                    case (see color parameter).
    """
    logging.info('Sanity check for the map...')
    assert color in self._df.columns, 'Color not valid. To see column names, type print(Obj.df.columns) .'
    assert (coord_type == 'equatorial' or coord_type == 'galactic'), 'Use only equatorial or galactic coordinates, please!'

    logging.info('Preparing data for the map...')
    color_label = color

    # clean color column
    temp_df = self.clean_column(color_label).df

    if coord_type == 'equatorial':
      lon_label = 'RAJ2000'
      lat_label = 'DEJ2000'
    if coord_type == 'galactic':
      lon_label = 'GLON'
      lat_label = 'GLAT'

    lon = temp_df[lon_label]
    lat = temp_df[lat_label]
    col = temp_df[color_label]

    # convert deg values to RA
    lon = coord.Angle(lon * u.deg)
    lon = lon.wrap_at(180 * u.deg).radian
    lat = coord.Angle(lat * u.deg).radian

    # build dataframe
    coord_df = pd.DataFrame({lon_label:lon, lat_label:lat, color_label:col})

    logging.info('Preparing the map...')
    fig, ax = plt.subplots(1, 1)
    ax = plt.axes(projection='mollweide')
    ax.grid(b=True)

    #if values are discrete, then plot a legend
    if color == 'CLASS1':
      sns.scatterplot(x=lon_label, y=lat_label, hue=color_label,
              data=coord_df, **kwargs)
      #ax.set_position(pos = [0.15, 0.2, 0.6, 0.6])
      ax.legend(loc='lower center', ncol=6)
    # else plot a colorbar
    elif color in self._df.columns and is_numeric_dtype(self._df[color_label]):
      scat = ax.scatter(lon, lat, c=col.tolist(), **kwargs)
      ax.set_xlabel(lon_label)
      ax.set_ylabel(lat_label)
      cbar = fig.colorbar(scat)
      cbar.set_label(color_label)
    # else draw with no colors
    else:
      if color is not None:
        # raise warning
        warnings.warn('Not valid value for color column!')
      ax.scatter(lon, lat, **kwargs)
    ax.set_title(title)

    logging.info('Map ready to be shown or saved!')
    show_plot(savefig=savefig, title=title)


  def classifier(self, predict_unassociated=False):
    """
    Generates a Decision Tree with the purpose to classify the unassociated sources of the catalog. 
    The categories of the sources are in the CLASS1 column. First, we map each category to a integer (integer encoding).
    Then, we divide the columns in feature (independent) variables and target (dependent) variables.
    The target is simply the CLASS1 column. The features are the rest of the columns (except the ones containing strings).
    Since the Decision Tree cannot operate with NaN values, we filled them all with the mean value of the column
    to which that value belongs.
    
    The data that generates the Decision Tree is made up of all the sources except the unassociated ones. 
    Decision Trees tend to overfit so we reduced its size and complexity (pruning). Plus, we removed all
    the categories populated by less than 5 sources so to improve the algorithm.

    :param predict_unassociated: if True, predicts the category of the unassociated sources
    """
    self.clean_column('CLASS1')

    #Integer encoding
    self._df['CLASS1'] = self._df['CLASS1'].map({'agn': 0,'bcu': 1,'bin': 2,'bll': 3,'css': 4,
               'fsrq': 5, 'gal': 6,'glc': 7,'hmb': 8,'lmb': 9,'nlsy1': 10,
               'nov': 11,'psr': 12,'pwn': 13,'rdg': 14,'sbg': 15,'sey': 16,
               'sfr': 17,'snr': 18,'spp': 19, 'ssrq': 20, 'unassociated': 21, 'unk': 22})

    #Remove categories that aren't very populated to make a better algorithm
    df_filtered = self._df.query('CLASS1 != 4 & CLASS1 != 6 & CLASS1 != 17 & CLASS1 != 20 & CLASS1 != 9 & CLASS1 != 2  & CLASS1 != 16 & CLASS1 != 11')
    logging.info('Underpopulated classes removed.')

    y = df_filtered['CLASS1'][df_filtered['CLASS1'] != 21]   #Target: the source category. we exclude the unassociated sources,
                                                             #because we will predict them based on the decision tree
    X = df_filtered[df_filtered['CLASS1'] != 21].select_dtypes(exclude='object')    #Features
    X = X.fillna(X.mean())     #replace missing data with the mean of the column
    
    # Create Decision Tree classifer 
    clf = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=10, min_samples_leaf=5, max_depth=5)   #Limit depth (aka prune tree) to avoid overfitting 
    logging.info('Generated the Decision Tree Classifier.')

    # Generate learning curves (see tutorial https://www.dataquest.io/blog/learning-curves-machine-learning/)
    train_sizes, train_scores, validation_scores = learning_curve(estimator = clf,
                                                                  n_jobs = -1,
                                                                  X = X,
                                                                  y = y, cv = 5, shuffle=True,
                                                                  train_sizes = np.linspace(1, 1000, dtype=int),
                                                                  scoring = 'accuracy')
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)

    clf = clf.fit(X, y)
    y_pred = clf.predict(X)

    logging.info("Accuracy: "+str(metrics.accuracy_score(y, y_pred)))     # Model Accuracy

    # Plot Learning curves
    logging.info('Started plotting the learning curves...')
    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Learning curves', fontsize = 18, y = 1.03)
    plt.legend()
    plt.savefig(output_path + '/' + 'Learning curves' + '.png')
    logging.info('See output folder for the learning curves!')

    if predict_unassociated:
      X_unassociated = self._df[self._df['CLASS1'] == 21].select_dtypes(exclude='object')
      X_unassociated = X_unassociated.fillna(X.mean())
      y_pred_unass = clf.predict(X_unassociated)
      counter = Counter(y_pred_unass)
      print(counter)

