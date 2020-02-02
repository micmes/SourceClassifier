import os
import pandas
from astropy.io import fits
from astropy.table import Table

def open_fits(path):
    """Opens a FITS file, given the path to the file, gets rid of multidimensional
    columns and returns a pandas dataframe. 
    """
    try:
        with fits.open(path) as hdul:
            t_astropy = Table(hdul[1].data)
            col1D = [col1D for col1D in t_astropy.colnames if len(t_astropy[col1D].shape) <= 1]
            data = t_astropy[col1D].to_pandas()
            return data
    except OSError as e:
        print(e)


if __name__ == '__main__':
    data_path = os.path.abspath('gll_psc_v21.fit')
    data = open_fits(data_path)