import os
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt

# requires setup.sh to run
source_root = os.environ["SOURCE_ROOT"]
output_path = source_root + '/output'

class Fermi_Dataset:
    """
    Class capable to perform data analysis and to visualize the given dataset in a graphical fashion.
    The Fermi_Dataset class is targeted at the 4FGL forth source catalog.
    """

    def __init__(self, handler):
        """
        Constructor. 
        The argument is the file handler for the FITS file containing the data.
        The handler is an HDUL file and the data is contained in the first extension. 
        We get rid of the multidimensional columns in order to use pandas.
        The attribute df is the dataframe originating from the HDUL.
        The attribute filtered_df is the same dataframe on which some filtering was performed. 
        """
        try:
            fits_data = handler[1].data
        except Exception as e:
            print(e)

        t_astropy = Table(fits_data)
        col1D = [col1D for col1D in t_astropy.colnames if len(t_astropy[col1D].shape) <= 1]
        data = t_astropy[col1D].to_pandas()

        self._df = data
        self._filtered_df = self._df.copy(deep=1)
    
    @property
    def df(self):
        return self._df

    @property
    def filtered_df(self):
        return self._filtered_df

    def filtering(self, df_condition):
        """
        Selects dataframe rows based on df_condition.
        Note that the filtering operation is always performed over the original data. 
        So, if you want to filter data on filtered data, you must unite all the conditions
        and perform the filtering only once.
        """
        self._filtered_df = self._df[df_condition]
        return self

    def columns(self):
        """
        Show column names
        """
        print(self._df.columns)

    def source_hist(self, colname, filter=False, title='Histogram', xlabel='x',
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
        if filter==True:
            plt.hist(self.filtered_df[colname], **kwargs)
        else:
            plt.hist(self.df[colname], **kwargs)
        

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        '''if savefig:
            plt.savefig('{}/{}'.format(output_path,title))
        else:
            plt.show()'''
        plt.savefig(title)
        return

    def galactic_map(self):
        """
        Plot a galactic map given sources"""


    def dist_models(self):
        """
        Plots the distribution of the 3 models of the sources.
        """
        self.source_hist('SpectrumType', title='Distribution of Models',
                            xlabel='Spectrum Model', ylabel='Number of sources', 
                            bins=3, histtype='bar')
        

if __name__ == '__main__':

    data_path = os.environ["SOURCE_ROOT"] + '/data/gll_psc_v21.fit'
    try:
        with fits.open(data_path) as hdul:
            data_4FGL = Fermi_Dataset(hdul)
    except OSError as e:
        print(e)
    
    #prove
    condition_blazar = data_4FGL.df['CLASS1'].str.match('(bll)|(BLL)')
    data_4FGL.filtering(condition_blazar)
    
    data_4FGL.filtering(data_4FGL.df['CLASS1'].str.match('(psr)|(PSR)'))
    
    data_4FGL.columns()
    '''
    data_4FGL.df['Energy_Flux100'] = data_4FGL.df['Energy_Flux100'].multiply(1e12)
    high_latitude_sources = data_4FGL[(abs(data_4FGL.df['GLAT'])>30)]
    '''
