# -*- coding: utf-8 -*-
"""  A function that performs simple Monte Carlo integration """

import pdb

# pdb.set_trace()

def drawStdNormalPoints(numSamples):
    from numpy import random
    from scipy.stats import norm
    uniformPoints = random.rand(numSamples)
    StdNormalPoints = norm.ppf(uniformPoints)
    return StdNormalPoints

def drawNormalPoints(numSamples,mu,sigma):
    StdNormalPoints = drawStdNormalPoints(numSamples)
    return mu + sigma*StdNormalPoints

def fnc2integrate(x,fncExpnt):
    if x < 0:
        fncValue = 0
    else:
        fncValue = x**fncExpnt 
    return fncValue

def calcFncValues(numSamples,normalPoints,fncExpnt):
    from numpy import zeros
    
    fncValuesArray = zeros((numSamples,1))
    
    idx = 0
    while idx < numSamples:
        fncValuesArray[idx] = fnc2integrate(normalPoints[idx],fncExpnt)
        idx = idx + 1        
        
    return fncValuesArray.T

def integrateFncValues(fncValueArray,numSamples):
    from numpy import sum 
    mcIntegral = sum(fncValueArray)/numSamples
    return mcIntegral

def calcAnalyticIntegral(sigma,fncExpnt):
    from scipy.special import gamma as gamma_fnc
    from math import sqrt
    from math import pi
    analyticIntegral = (1/(sqrt(2*pi)*sigma)) \
                        *0.5*gamma_fnc(0.5*(fncExpnt+1)) \
                        *(2*sigma**2)**(0.5*(fncExpnt+1))
#    pdb.set_trace()
    return analyticIntegral
    
def computeRmse(analyticIntegral,mcIntegral):
    from numpy import sqrt, mean
    
#    print("In computeRmse")

    return sqrt(mean((mcIntegral - analyticIntegral) ** 2))    
    
def computeRmseN(numSamples):
    from numpy import zeros, arange

#    print("In computeRmseN")
   
    mu = 0
    sigma = 1
    fncExpnt = 4
    numExperiments = 1000

    mcIntegral = zeros(numExperiments)

    analyticIntegral =  calcAnalyticIntegral(sigma,fncExpnt)   
    print "Analytic calculation of integral = %s" % analyticIntegral

    for idx in arange(numExperiments):

        normalPoints = drawNormalPoints(numSamples,mu,sigma)
#    print"NormalPoints = %s" % normalPoints
    
        fncValuesArray = calcFncValues(numSamples,normalPoints,fncExpnt)    
#    print"Function values = %s" % fncValuesArray  

#        pdb.set_trace()
    
        mcIntegral[idx] = integrateFncValues(fncValuesArray,numSamples)
#        print "Monte Carlo estimate = %s" % mcIntegral[idx]    
        
    
    rmse = computeRmse(analyticIntegral,mcIntegral)
    print "RMSE of Monte Carlo estimate = %s" % rmse
    
    return rmse    
    
def main():
    from numpy import zeros, arange, sqrt
    import matplotlib.pyplot as plt
    
    numNValues = 10

    rmseNValues = zeros(numNValues)    
    numSamplesN = zeros(numNValues)

    
    for idx in arange(numNValues):    
        numSamplesN[idx] = 2**(idx+2)
        print "numSamplesN = %s" % numSamplesN[idx]
        rmseNValues[idx] = computeRmseN(numSamplesN[idx])
    
    theoryError = 100.0/sqrt(numSamplesN)    
    
    plt.clf()
#    plt.subplot(221)
    plt.loglog(numSamplesN, rmseNValues, label='MC Error')    
    plt.loglog(numSamplesN, theoryError, label='Theory (1/sqrt(N))')
    plt.legend()
    plt.xlabel('Number of sample points')
    plt.ylabel('Root-mean-square error')
    plt.show()    
    
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()

    