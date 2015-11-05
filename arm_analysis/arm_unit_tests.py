# -*- coding: utf-8 -*-
"""
   Unit tests related to the analysis of ARM observations. 
"""

import unittest
from arm_utilities import findTruncNormalRoots

class testFindTruncNormalRoots(unittest.TestCase):
    
  def test_findTruncNormalRoots(self):
    """Tests whether our root finder for a truncated normal
    works in the case of a standard normal truncated at 0"""

    from numpy import sqrt
    from math import pi

    # These are the analytic values of mean and variance of a standard normal
    # truncated on the left at 0. 
    truncMean = 2.0 / sqrt(2*pi)
    truncVarnce = 1.0 - 4.0/(2.0*pi)

    muInit = truncMean 
    sigmaInit = sqrt(truncVarnce)

    minThreshRefl = 0

    mu, sigma = findTruncNormalRoots(truncMean,truncVarnce,
                                     muInit,sigmaInit,
                                     minThreshRefl)

    self.assertAlmostEqual(mu, 0, places=10)
    self.assertAlmostEqual(sigma, 1, places=10)
    
    

if __name__ == '__main__':
#    unittest.main()    
    try:
        unittest.main()
    except SystemExit as inst:
        if inst.args[0] is True: # raised by sys.exit(True) when tests failed
            raise