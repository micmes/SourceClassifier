import unittest

# make sure a __init__.py file exist in the import folder
from sourceclassifier.sourcehist import sourcehist

class testSourceClassifier(unittest.TestCase):

    def test_sourcehist(self):
        """
        Test the sourcehist function"""
        x = [1,2,3,4,5,4,2,3,4,2,1,5,6]

        sourcehist(x, title='plt hist test', xlabel='values', ylabel='occurrences')

if __name__ == '__main__':
	unittest.main()
