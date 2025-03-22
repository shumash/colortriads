import os
import unittest

def getAbsoluteTestdataPath(paths):
    '''
    Converts a list of path components, e.g. ['util', 'img.png']
    to an absolute path inside testdata/python of this repo.
    '''
    modulepath = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(
        os.path.join(modulepath, os.pardir, os.pardir, 'testdata', 'python'),
        *paths)

class UtilTestCase(unittest.TestCase):
    def assertColorsEqual(self, color0, color1, msg='', delta=0.001):
        message = '%sExpected %s == %s' % (msg, str(color0), str(color1))
        for i in range(3):
            self.assertAlmostEqual(color0[i], color1[i], msg=message, delta=delta)

    def assertVectorsAlmostEqual(self, vec0, vec1):
        self.assertEqual(len(vec0), len(vec1))
        for i in range(len(vec0)):
            self.assertAlmostEqual(vec0[i], vec1[i])