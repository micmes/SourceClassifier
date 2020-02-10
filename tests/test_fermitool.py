import unittest
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

		TEST_DF.source_hist('SpectrumType', title='TEST_Spectralmodel_withsourcehist', savefig=True)

	def test_cleanclasses(self):
		"""
		Test the clean classes function.
		"""
		cleaned_dataset = TEST_DF.clean_nan('CLASS1').clean_classes()
		self.assertTrue(cleaned_dataset.df['CLASS1'].str.islower().all(), 'classes are not all lowercase')

	def test_cleanNaN(self):
		"""
		Test the clean_nan function.
		"""
		cleanedNaN_dataset = TEST_DF.clean_nan('Signif_Avg')
		self.assertFalse(cleanedNaN_dataset.df['Signif_Avg'].isnull().values.any())

	def test_filtering(self):
		"""
		Test filtering function
		"""
		TEST_DF.df['RAJ2000']
		TEST_DF.filtering(TEST_DF.df['RAJ2000'] > 180).df['RAJ2000']

	def test_galactic_map(self):
		"""
		Test the galactic map function
		"""
		TEST_DF.galactic_map(title='TEST_all_data_CLASS', color='CLASS1',
							 savefig=True, palette='prism_r', marker='x',
							 alpha=0.5)
		TEST_DF.galactic_map(title='TEST_all_data_DEC', color='DEJ2000',
							 savefig=True, palette='YlGn', marker='x',
							 alpha=0.1)

		TEST_DF.filtering(TEST_DF.df['DEJ2000'] > 0).galactic_map(title='TEST_Only_positive_dec', color='CLASS1',
																  savefig=True)
		TEST_DF.filtering(TEST_DF.df['DEJ2000'] > 0).galactic_map(coord_type='galactic', title='TEST_Only_positive_dec_but_galactic',
																  color='CLASS1',
																  savefig=True)

	def test_energyflux_map(self):
		"""
		Test the energyflux_map method.
		"""
		TEST_DF.energyflux_map(savefig=True)
	
	
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
