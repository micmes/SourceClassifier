import os
import pandas
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt

class Fermi_Dataset:
    """
    Class capable to perform data analysis and to visualize the given dataset in a graphical fashion.
    The Fermi_Dataset class is targeted at the 4FGL forth source catalog.
    """

    def __init__(self, handler):
        """
        Constructor. 
        The argument is the file handler for the FITS file containing the data. 
        """
        self._df = self.fits2df(handler)
    
    @property
    def df(self):
        return self._df
    

    def fits2df(self, hdul):
        """
        Given the hdul of the FITS file, this method gets rid of multidimensional
        columns and returns a pandas dataframe. 
        """
        try:
            fits_data = hdul[1].data
        except Exception as e:
            print(e)

        t_astropy = Table(fits_data)
        col1D = [col1D for col1D in t_astropy.colnames if len(t_astropy[col1D].shape) <= 1]
        data = t_astropy[col1D].to_pandas()
    
        return data


    def select_data(self, df_condition):
        """
        Selects dataframe rows based on df_condition.
        """
        return self._df[df_condition]

    def sourcehist(self, x, title='Histogram', xlabel='x', ylabel='y',
               **kwargs):
        """
        This function provides a histogram plot given a single array in
        input. Most of the features are inherited from the matplotlib hist
        function.
        :x: an array of values
        :title: the title of the histogram shown in the plot
        :xlabel: x label shown in the plot
        :ylabel: y label shown in the plot
        :kwargs: the same parameters of the plt.hist function
        :return: a histogram plot of values.
        """

        plt.hist(x, **kwargs)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(title)
        
    
    def dist_models(self, dataframe=None):
        if dataframe == None:
            dataframe = self._df
        self.sourcehist(dataframe['SpectrumType'], title='Distribution of Models', 
                            xlabel='Spectrum Model', ylabel='Number of sources', 
                            bins=3, histtype='bar')
        

if __name__ == '__main__':

    data_path = os.path.abspath('../data/gll_psc_v21.fit')
    try:
        with fits.open(data_path) as hdul:
            data_4FGL = Fermi_Dataset(hdul)
    except OSError as e:
        print(e)
    
    #prove
    condition_blazar = data_4FGL.df['CLASS1'].str.match('(bll)|(BLL)')
    filtered_blazars = data_4FGL.select_data(condition_blazar)
   #print(filtered_blazars['CLASS1'])
    
    data_4FGL.df['Energy_Flux100'] = data_4FGL.df['Energy_Flux100'].multiply(1e12)
    high_latitude_sources = data_4FGL.select_data(abs(data_4FGL.df['GLAT'])>30)
    data_4FGL.dist_models()
