import os
from astropy.io import fits
from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
from matplotlib import pyplot as plt

# requires setup.sh to run
source_root = os.environ["SOURCE_ROOT"]
output_path = source_root + '/output'


class Fermi_Dataset:
	"""
	Class capable to perform data analysis and to visualize the given dataset in a graphical fashion.
	The Fermi_Dataset class is targeted at the 4FGL forth source catalog.
	"""

	def __init__(self, data):
		"""
		Constructor.
		The argument is the file handler for the FITS file containing the data.
		The handler is an HDUL file and the data is contained in the first extension.
		We get rid of the multidimensional columns in order to use pandas.
		The attribute df is the dataframe originating from the HDUL.
		"""
		self._df = data

	@property
	def df(self):
		return self._df

	def filtering(self, df_condition):
		"""
		Selects dataframe rows based on df_condition.
		Note that the filtering operation is always performed over the original data.
		So, if you want to filter data on filtered data, you must unite all the conditions
		and perform the filtering only once.
		"""
		return Fermi_Dataset(self._df[df_condition])

	def col(self, colname):
		"""
		Return the content of a given column
		"""
		return self._df[colname]

	def columns(self):
		"""
		Return column names
		"""
		return self._df.columns

	def source_hist(self, colname, title='Histogram', xlabel='x',
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
		plt.figure()
		plt.hist(self.df[colname], **kwargs)
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		if savefig:
			# create dir if it doesn't exist
			if not os.path.isdir(output_path):
				os.makedirs(output_path)
			plt.savefig(output_path + '/' + title + '.png')
		else:
			plt.show()
		return

	def galactic_map(self, title='Galactic Map', savefig=False, c=None,
					 colorbar=False, **kwargs):
		"""
		Plot a galactic map given sources. We're assuming that the right
		ascension and the declination columns are labeled by 'RAJ2000' and
		'DEJ2000' respectively.
		:title:the title of the histogram shown in the plot (str)
		:savefig: choose whether to save the fig or not (in the output folder)
		:c: colours in the scatter plot. Every value that satisfies the
		'matplot.pyplot.scatter' conditions is accepted; in addition,
		string values that match the column names give a gradient according
		to values in the column itself.
		:kwargs: set the points parameters according to 'matplotlib.pyplot.scatter' module
		"""

		d = self._df

		# check whether 'c' is a column name. If yes, use the values in
		# that column to color the scatter plot
		if c in self.columns():
			c = d[c]

		ra = coord.Angle(d['RAJ2000'] * u.degree)
		ra = ra.wrap_at(180 * u.degree)
		dec = coord.Angle(d['DEJ2000'] * u.degree)

		fig = plt.figure(figsize=(8, 6))
		ax = fig.add_subplot(111, projection="mollweide")
		ax1 = ax.scatter(ra.radian, dec.radian, c = c, **kwargs)

		if colorbar:
			fig.colorbar(ax1)
		plt.title(title)

		if savefig:
			# create dir if it doesn't exist
			if not os.path.isdir(output_path):
				os.makedirs(output_path)
			fig.savefig(output_path + '/' + title + '.png')
		else:
			fig.show()
		return

	def dist_models(self):
		"""
		Plots the distribution of the 3 models of the sources.
		"""
		self.source_hist('SpectrumType', title='Distribution of Models',
						 xlabel='Spectrum Model', ylabel='Number of sources',
						 bins=3, histtype='bar')


if __name__ == '__main__':

	# import fits file
	data_path = os.environ["SOURCE_ROOT"] + '/data/gll_psc_v21.fit'

	try: # does the file exist? If yes
		with fits.open(data_path) as hdul:
			fits_data = hdul[1].data
	except OSError as e:
		print(e)

	# remove multicolumn
	t_astropy = Table(fits_data)
	col1D = [col1D for col1D in t_astropy.colnames if len(t_astropy[col1D].shape) <= 1]
	data = t_astropy[col1D].to_pandas()

	# define an istance
	data_4FGL = Fermi_Dataset(data)

	# prove
	condition_blazar = data_4FGL.df['CLASS1'].str.match('(bll)|(BLL)')
	data_4FGL.filtering(condition_blazar)

	#data_4FGL.filtering(data_4FGL.df['CLASS1'].str.match('(psr)|(PSR)'))

	data_4FGL.columns()
	data_4FGL.col('DEJ2000')
	'''
	data_4FGL.df['Energy_Flux100'] = data_4FGL.df['Energy_Flux100'].multiply(1e12)
	high_latitude_sources = data_4FGL[(abs(data_4FGL.df['GLAT'])>30)]
	'''
