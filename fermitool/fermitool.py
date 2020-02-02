import os
import pandas
from astropy.io import fits
from astropy.table import Table

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
    print(filtered_blazars['CLASS1'])