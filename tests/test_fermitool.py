import unittest
import pytest
# make sure a __init__.py file exist in the import folder
# requires setup.sh to run

from astropy.io import fits
from astropy.table import Table

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

	def test_def_column(self):

		def add(a,b):
			return a + b

		dfA = TEST_DF.def_column(['RAJ2000','DEJ2000'], add, newcolname='Sum').df
		dfB = TEST_DF.df
		dfB['Sum_for_sure'] = dfB['RAJ2000'] + dfB['DEJ2000']
		check = np.where(dfA['Sum'] == dfB['Sum_for_sure'], True, False)
		self.assertTrue(all(check))

		def add3(a, b, c):
			return a + b + c

		dfA3 = TEST_DF.def_column(['RAJ2000', 'DEJ2000', 'GLAT'], add3, newcolname='Sum').df
		dfB3 = TEST_DF.df
		dfB3['Sum_for_sure'] = dfB3['RAJ2000'] + dfB3['DEJ2000'] + dfB3['GLAT']
		check = np.where(dfA3['Sum'] == dfB3['Sum_for_sure'], True, False)
		self.assertTrue(all(check))

	def test_sourcehist(self):
		"""
		Test the sourcehist function
		"""

		TEST_DF.source_hist('RAJ2000', title='TEST_Ra(J2000)',
							xlabel='bins', ylabel="occurrences",
							savefig=True)


		TEST_DF.filtering(abs(TEST_DF.df['GLAT']) > 30).source_hist('RAJ2000',
																	title='TEST_RA(J2000) with condition',
																	xlabel='bins', ylabel='occurrences',
																	savefig=True)

		# TEST_DF.source_hist('SpectrumType', title='TEST_Spectralmodel_withsourcehist', savefig=True)

	def test_cleanclasses(self):
		"""
		Test the clean classes function.
		"""
		cleaned_dataset = TEST_DF.remove_nan_rows('CLASS1').clean_column('CLASS1')
		self.assertTrue(cleaned_dataset.df['CLASS1'].str.islower().all(), 'classes are not all lowercase')

	def test_cleanNaN(self):
		"""
		Test the clean_nan function.
		"""
		cleanedNaN_dataset = TEST_DF.remove_nan_rows('Signif_Avg')
		self.assertFalse(cleanedNaN_dataset.df['Signif_Avg'].isnull().values.any())

	def test_filtering(self):
		"""
		Test filtering function
		"""
		print(TEST_DF.df['RAJ2000'])
		print(TEST_DF.filtering(TEST_DF.df['RAJ2000'] > 180).df['RAJ2000'])

	def test_galactic_map(self):
		"""
		Test the galactic map function
		"""
		TEST_DF.galactic_map(title='TEST_all_data_CLASS', color='CLASS1',
							 savefig=True, palette='YlGn_r', marker='x',
							 alpha=0.5)
		TEST_DF.galactic_map(title='TEST_all_data_DEC', color='DEJ2000',
							 savefig=True, cmap='YlGn', marker='x',
							 alpha=0.1)

		TEST_DF.filtering(TEST_DF.df['DEJ2000'] > 0).galactic_map(title='TEST_Only_positive_dec', color='CLASS1',
																  savefig=True)
		TEST_DF.filtering(TEST_DF.df['DEJ2000'] > 0).galactic_map(coord_type='galactic', title='TEST_Only_positive_dec_but_galactic',
																  color='CLASS1',
																  savefig=True)
	
	def test_dist_models(self):
		"""
		Test the dist_models method.
		"""
		TEST_DF.dist_models(savefig=True)
	
	
	def test_plot_spectral_param(self):
		"""
		Test the plot spectral param method.
		"""
		TEST_DF.filtering(TEST_DF.df['Signif_Avg']>=30).plot_spectral_param(title='TEST_spectral_param only significant sources',savefig=True)
	
	def test_dist_variability(self):
		"""
		Test the dist_varibility_index method.
		"""
		TEST_DF.dist_variability(savefig=True)

	def test_compare_variability(self):
		"""
		Test the plot of the compare_variability method.
		"""
		TEST_DF.compare_variability(savefig=True)																		

if __name__ == '__main__':
	unittest.main()
