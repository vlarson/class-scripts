# -*- coding: utf-8 -*-
"""  A library of home-made functions for the purpose
    of performing Monte Carlo integration """

import pdb

# pdb.set_trace()

def drawStdNormalPoints(numSamples):
    from numpy import random
    from scipy.stats import norm

    uniformPoints = random.rand(numSamples)
    StdNormalPoints = norm.ppf(uniformPoints)

    return StdNormalPoints

def calcFncValues(numSamples,fncDim,samplePoints,fncIntegrand,*args):
    from numpy import zeros

#    pdb.set_trace()     

    # Assertion check: see if samplePoints has numSamples rows and fncDim columns
    if samplePoints.shape[0] != numSamples:
        print "Error: normalPoints does not have numSample rows in function calcFncValues."
    if (samplePoints.ndim == 1):
        if (fncDim != 1):
            print "Error: normalPoints has 1 column but fncDim != 1 in function calcFncValues."
    if (samplePoints.ndim > 1):
        if (samplePoints.shape[1] != fncDim):
            print "Error: normalPoints does not have fncDim columns in function calcFncValues."    

#    fncValuesArray = zeros((numSamples,fncDim))
    fncValuesArray = zeros(numSamples)
    
    idx = 0
    while idx < numSamples:
        # This should grab the idx'th row even if normalPoints is a 2D array
        fncValuesArray[idx] = fncIntegrand(samplePoints[idx], *args)
        idx = idx + 1        
        
    return fncValuesArray.T
    
def integrateFncValues(fncValueArray,numSamples):
    from numpy import sum 

    mcIntegral = sum(fncValueArray)/numSamples

    return mcIntegral
    
def computeRmse(analyticIntegral,mcIntegral):
    from numpy import sqrt, mean

    return sqrt(mean((mcIntegral - analyticIntegral) ** 2))    
    