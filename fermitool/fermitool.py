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
  The constructor takes a DataFrame as an argument.
  The Fermi_Dataset class is targeted at the 4FGL forth source catalog: all the column names are 
  those of the 4FGL. We therefore suggest to take a look at the following link <https://arxiv.org/abs/1902.10045>
  in order to get familiar with the data.
  In this class, some method are totally generic and allow to draw histograms and scatter plots,
  filtering data depending on user preferences. In addition, we built some methods in order to
  reproduce the results presented on 4FGL catalog description (linked above). For further details,
  we refer to the description of the methods below.
  """

  def __init__(self, data):
    """
    Constructor.

    :param data: pandas dataframe containing the data
    """
    self._df = data

  @property
  def df(self):
    """
	  The property prevents the user to accidentally modify the instance.
	  """
    return self._df


  def def_column(self, colnames, myfunc, newcolname='TEMP'):
    """
    This method provides a tool to create a new column applying a function
    that takes two or more columns as parameters. This is useful for example
    in evaluating geometric mean when the user wants to draw the error radii histogram.
    Please notice that the original df remains unchanged. This method provides a
    new instance of the Fermi_Dataset class.

    :param colnames: the column that are given as inputs.
    :param myfunc: the function that takes the columns as inputs. Please be
    careful to set the number of parameters equal to the number of values given
    as input
    :param newcolname: the column name of the function output.

    :return: a new Fermi_Dataset object.
    """
    # please notice: the * inside the function arguments split the numpy matrix
    # into a list of arrays, so that 2 or more parameters are given in input.
    new_df = self.df
    new_df[newcolname] = myfunc(*self.df[colnames].values.T)
    return Fermi_Dataset(new_df)

  def filtering(self, df_condition):
    """
    Selects dataframe rows based on df_condition. Returns a new object with the filtered data.

    :param df_condition: The condition based on which you filter the dataframe.
    Make sure that the condition is written conformly to the pandas module:
    more specifically, 'df_condition' is the same condition of pandas.DataFrame.loc,
    so that a new instance of Fermi_Dataset is based on the new df:

    .. code-block:: python

       new_df = df.loc[df_condition]

    View https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html
    for further explanation.

    :return: a new Fermi_Dataset object.
    """
    try:
      # Notice that this indexing return a copy of the original df, so the
      # original instance remains unchanged
      new_df = self.df.loc[df_condition]
      return Fermi_Dataset(new_df)
    except Exception as e:
      print('Oops! Give me a valid condition', e)
      raise # for pytest?

  def clean_column(self, colname):
    """
    Removes extra spaces and replaces empty values with '_unassociated_'. This operation is useful
    for lowering all the characters in a given column of the dataframe. The empty rows
    are replaced plots, when we don't need to distinguish between associated and
    identified sources (in CAPS).
    Please notice that the original df remains unchanged. This method provides a
    new instance of the Fermi_Dataset class.

    :param colname: the column that is given as input.

    :return: a new instance of Fermi_Dataset class.
    """
    assert is_string_dtype(self.df[colname]), 'Column is not string type. Please assert ' \
                                              'that the given column is string type'

    new_df = self.df
    new_df[colname] = new_df[colname].apply(lambda x: x.strip().lower())
    new_df[colname] = new_df[colname].replace('', 'unassociated')
    logging.info('Column cleaned successfully.')

    return Fermi_Dataset(new_df)


  def remove_nan_rows(self, collist):
    """
    Given the column name, remove the rows of the dataframe where the values
    of that column are NaN.
    Please notice that the original df remains unchanged. This method provides a
    new instance of the Fermi_Dataset class.

    :param collist: list of column to which remove the nan values.

    :return: a new instance of Fermi_Dataset class.
    """

    try:
      new_df = self.df
      new_df = new_df.dropna(subset=collist)
      logging.info('Cleaned NaN rows from {} columns.'.format(collist))

      return Fermi_Dataset(new_df)

    except Exception as e:
      print("Oops! {} is not valid. Please notice that if a single column is given,"
            "it should be written as an iterable (for example: ['RAJ2000'] and not "
            "'RAJ2000'). To see column names, type print(Obj.df.columns)".format(collist))
      raise

  def source_hist(self, colname, title='Histogram', savefig=False, xlog=False, ylog=False, **kwargs):
    """
    This method provides a histogram plot given a single array in
    input. Most of the features are inherited from the matplotlib hist
    function.

    :param colname:  The name of the column to plot. The column must be
    numeric.
    :param title:  the title of the histogram shown in the plot
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
    plt.xlabel(colname)
    plt.ylabel('Counts')

    if xlog:
      plt.xscale('log')
    if ylog:
      plt.yscale('log')

    logging.info('Histogram ready to be shown or saved!')
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
				  take its place.
	:param **kwargs: other parameters passed directly to
					'seaborn.scatterplot' or 'matplotlib.pyplot.scatter' depending on the
					case (see color parameter).

	"""
    logging.info('Sanity check for the map...')
    assert color in self.df.columns, 'Color not valid. To see column names, type print(Obj.df.columns) .'
    assert (coord_type == 'equatorial' or coord_type == 'galactic'), 'Not valid value given for coord_type. Try with' \
                                                                     '"equatorial" or "galactic".'

    logging.info('Preparing data for the map...')

    color_label = color

    if coord_type == 'equatorial':
      lon_label = 'RAJ2000'
      lat_label = 'DEJ2000'
    if coord_type == 'galactic':
      lon_label = 'GLON'
      lat_label = 'GLAT'

    lon = self.df[lon_label]
    lat = self.df[lat_label]
    col = self.df[color_label]

    # convert deg values to RA
    lon = coord.Angle(lon * u.deg)
    lon = lon.wrap_at(180 * u.deg).radian
    lat = coord.Angle(lat * u.deg).radian

    # build dataframe
    coord_df = pd.DataFrame({lon_label: lon, lat_label: lat, color_label: col})

    logging.info('Preparing the map...')
    fig, ax = plt.subplots(1, 1)
    ax = plt.axes(projection='mollweide')
    ax.grid(b=True)

    # if values are discrete, then plot a legend
    if color == 'CLASS1':
      sns.scatterplot(x=lon_label, y=lat_label, hue=color_label,
                      data=coord_df, **kwargs)
      # ax.set_position(pos = [0.15, 0.2, 0.6, 0.6])
      ax.legend(loc='lower center', ncol=6)
    # else plot a colorbar
    elif color in self.df.columns and is_numeric_dtype(self.df[color_label]):
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


  def dist_models(self, title='Distribution of Spectral Models', savefig=False, **kwargs):
    """
    Plots the bar chart of the 3 models of the sources.

    :param title: title of the plot
    :param savefig: if True, save figure in output directory
    :param kwargs: other kwargs are passed directly to matplotlib.pyplot.bar
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
    data1 = self.df[['PLEC_Index', 'PLEC_Expfactor', 'CLASS1']]
    data2 = self.df[['LP_Index', 'LP_beta', 'CLASS1']]

    logging.info('Preparing the plot...')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.scatterplot(x='PLEC_Index', y='PLEC_Expfactor', hue='CLASS1', data=data1, ax=ax1)
    sns.scatterplot(x='LP_Index', y='LP_beta', hue='CLASS1', data=data2, ax=ax2)

    ax1.legend().set_visible(False)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()

    logging.info('Spectral parameters plot ready to be shown or saved!')
    show_plot(savefig=savefig, title=title)


  def compare_variability(self, title='Comparison of Varibility index for 2 month intervals and that for 12 months',
                          savefig=False, **kwargs):
    """
    Plot Variability Index 2 month vs Variability Index 12 month.

    :param title: title of the plot
    :param savefig: if True, save figure in the output directory
    :param kwargs: kwargs of matplotlib.pyplot.scatter module
    """
    logging.info('Preparing data for variability plot...')
    x = self.df['Variability_Index']
    y = self.df['Variability2_Index']

    logging.info('Preparing the variability plot...')
    plt.figure()
    plt.scatter(x, y, marker='+', **kwargs)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('12 month')
    plt.ylabel('2 month')

    logging.info('Variability plot ready to be shown or saved!')
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
    df_clean = self.clean_column('CLASS1').df

    #Integer encoding
    df_clean['CLASS1'] = df_clean['CLASS1'].map({'agn': 0, 'bcu': 1, 'bin': 2, 'bll': 3, 'css': 4,
               'fsrq': 5, 'gal': 6,'glc': 7,'hmb': 8,'lmb': 9,'nlsy1': 10,
               'nov': 11,'psr': 12,'pwn': 13,'rdg': 14,'sbg': 15,'sey': 16,
               'sfr': 17,'snr': 18,'spp': 19, 'ssrq': 20, 'unassociated': 21, 'unk': 22})

    #Remove categories that aren't very populated to make a better algorithm
    df_filtered = df_clean.query('CLASS1 != 4 & CLASS1 != 6 & CLASS1 != 17 & CLASS1 != 20 & CLASS1 != 9 & CLASS1 != 2  & CLASS1 != 16 & CLASS1 != 11')
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
    plt.ylim(0,1)
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Decision Tree curve', fontsize = 18, y = 1.03)
    plt.legend()
    plt.savefig(output_path + '/' + 'decisiontree' + '.png')
    logging.info('See output folder for the learning curves!')

    if predict_unassociated:
      X_unassociated = self.df[self.df['CLASS1'] == 21].select_dtypes(exclude='object')
      X_unassociated = X_unassociated.fillna(X.mean())
      y_pred_unass = clf.predict(X_unassociated)
      counter = Counter(y_pred_unass)
      print(counter)

