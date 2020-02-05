import unittest
from unittest.mock import patch

# make sure a __init__.py file exist in the import folder
# requires setup.sh to run
from fermitool.fermitool import *

# import fits file
data_path = os.environ["SOURCE_ROOT"] + '/data/gll_psc_v21.fit'
try:  # does it exists? If yes
	with fits.open(data_path) as hdul:
		fits_data = hdul[1].data
except OSError as e:
	print(e)

# remove multicolumn
t_astropy = Table(fits_data)
col1D = [col1D for col1D in t_astropy.colnames if len(t_astropy[col1D].shape) <= 1]
data = t_astropy[col1D].to_pandas()

# define an istance
TEST_DF = Fermi_Dataset(data)

class testFermiTool(unittest.TestCase):

	def test_sourcehist(self):
		"""
		Test the sourcehist function
		"""
		TEST_DF.source_hist('RAJ2000', title='TEST_Ra(J2000)',
							xlabel='bins', ylabel="occurrences",
							savefig=True)

		# try filtering
		TEST_DF.filtering(abs(TEST_DF.df['GLAT']) > 30).source_hist('RAJ2000',
																	title='TEST_RA(J2000) with condition',
																	xlabel='bins', ylabel='occurrences',
																	savefig=True)

	def test_filtering(self):
		"""
		Test filtering function
		"""
		TEST_DF.col('RAJ2000')
		TEST_DF.filtering(TEST_DF.df['RAJ2000'] > 180).col('RAJ2000')

	def test_galactic_map(self):
		"""
		Test the galactic map function
		"""
		TEST_DF.galactic_map(title='TEST_all_data', savefig=True)
		TEST_DF.filtering(TEST_DF.df['DEJ2000'] > 0).galactic_map(title='TEST_Only_positive_dec',
																  savefig=True, c='DEJ2000', colorbar=True)
		TEST_DF.filtering(TEST_DF.df['DEJ2000'] > 0).galactic_map(coord_type='galactic', title='TEST_Only_positive_dec_but_galactic',
																  savefig=True, c='DEJ2000', colorbar=True)
	def test_plot_spectral_param(self):
		"""
		Test the plot spectral param method.
		"""
		TEST_DF.filtering(TEST_DF.df['Signif_Avg']>=30).plot_spectral_param(title='TEST_spectral_param only significant sources',savefig=True)
	
	def test_compare_variability(self):
		"""
		Test the plot of the compare_variability method.
		"""
		TEST_DF.compare_variability(savefig=True)																		

if __name__ == '__main__':
	unittest.main()
