import os
from astropy.io import fits
from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from pandas.api.types import is_numeric_dtype, is_string_dtype
import pandas as pd
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

    :param data:  pandas dataframe containing the data
    """
    self._df = data


  @property
  def df(self):
    """
    DataFrame which contains the Fermi data.
    """
    return self._df

  
  def filtering(self, df_condition):
    """
    Selects dataframe rows based on df_condition. Returns an object with the filtered data.

    :param df_condition: The condition based on which you filter the dataframe.
                          Make sure that the condition is written conformly to the pandas module, as follows:
                          Fermi_Dataset_Instance.df['Column_Name'] ><= value
    """
    try:
      return Fermi_Dataset(self._df[df_condition])
    except Exception as e:
      print('Oops! Give me a valid condition', e)    
      raise


  def clean_column(self, col):
    """
    Removes extra spaces and lowers all the characters in the CLASS1 column of the dataframe.
    The empty rows are replaced with _unassociated_.
    This operation is useful for plots, when we don't need to distinguish between associated and identified sources(in CAPS).
    """
    if is_string_dtype(self._df[col]):
      #self._df[col] = self._df[col].apply(lambda x: x.strip().lower())
      #self._df[col] = self._df[col].replace('', 'unidentified')
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

    :param colname:  The name of the column to plot (str)
    :param filter:  Boolean value. If true, plot data from filtered_df
    :param title:  the title of the histogram shown in the plot (str)
    :param xlabel:  x label shown in the plot (str)
    :param ylabel:  y label shown in the plot (str)
    :param savefig:  choose whether to save the fig or not
    :param xlog: if True, set xscale to log
    :param ylog: if True, set yscale to log
    :param kwargs:  other parameters are passed to matplotlib.pyplot.hist
    module (str)
    """
    assert colname in self._df.columns, 'Column name not valid. To see column names, try print(Obj.df.columns) .' 
    logging.info('Preparing the histogram...')

    plt.figure()
    plt.hist(self.df[colname], **kwargs)
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
    lon = coord.Angle(lon * u.degree)
    lon = lon.wrap_at(180 * u.degree).radian
    lat = coord.Angle(lat * u.degree).radian

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
    The accuracy of the tree is calculated splitting the data into the training set and validation set.

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
    
    y = df_filtered['CLASS1'][df_filtered['CLASS1'] != 21]   #Target: the source category. we exclude the unassociated sources,
                                                             #because we will predict them based on the decision tree
    X = df_filtered[df_filtered['CLASS1'] != 21].select_dtypes(exclude='object')    #Features
    X = X.fillna(X.mean())     #replace missing data with the mean of the column
    
    #Split dataset in train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1) # 80% training and 20% test
    
    # Create Decision Tree classifer 
    clf = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=10, min_samples_leaf=5, max_depth=5)   #Limit depth (aka prune tree) to avoid overfitting
    clf = clf.fit(X_train, y_train)
    #Generate learning curves (see tutorial https://www.dataquest.io/blog/learning-curves-machine-learning/)
    train_sizes, train_scores, validation_scores = learning_curve(estimator = clf,
                                                                  X = X,
                                                                  n_jobs = -1,
                                                                  y = y, cv = 7, shuffle=True,
                                                                  train_sizes = np.linspace(0.01, 1.0, 50),
                                                                  scoring = 'accuracy')
    train_scores_mean = train_scores.mean(axis=1)
    validation_scores_mean = validation_scores.mean(axis=1)

    y_pred = clf.predict(X_test)    #Predict the response for test dataset
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))     # Model Accuracy

    # Plot Learning curves
    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Learning curves', fontsize = 18, y = 1.03)
    plt.legend()
    plt.savefig(output_path + '/' + 'Learning curves' + '.png')

    if predict_unassociated:
      X_unassociated = self._df[self._df['CLASS1'] == 21].select_dtypes(exclude='object')
      X_unassociated = X_unassociated.fillna(X.mean())
      y_pred_unass = clf.predict(X_unassociated)    
      counter = Counter(y_pred_unass)
      print(counter)

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
  print(data_4FGL.df.columns)
  print(data_4FGL.df['GLON'])
  print(data_4FGL.df['GLAT'])

  print(extended_4FGL.df.columns)
  
  
  data_4FGL.classifier()
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