import unittest

# make sure a __init__.py file exist in the import folder
# requires setup.sh to run
from fermitool.fermitool import *

# requires setup.sh to run
source_root = os.environ["SOURCE_ROOT"]
data_path = source_root + '/data/gll_psc_v21.fit'

try:
    with fits.open(data_path) as hdul:
        TEST_DF = Fermi_Dataset(hdul)
except OSError as e:
    print(e)

class testFermiTool(unittest.TestCase):

    def test_sourcehist(self):
        """
        Test the sourcehist function"""
        print(TEST_DF.df.columns)
        TEST_DF.source_hist('RAJ2000', title='Ra(J2000)',
                           xlabel='bins', ylabel="occurrences")

        # try filtering
        TEST_DF.filtering(abs(TEST_DF.df['GLAT'])>30)
        TEST_DF.source_hist('RAJ2000', filter=True,
                            title='RA(J2000) with condition',
                            xlabel='bins', ylabel='occurrences')

if __name__ == '__main__':
    unittest.main()
