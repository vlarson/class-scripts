# -*- coding: utf-8 -*-
"""  A function that performs simple Monte Carlo integration """

import pdb

# pdb.set_trace()

def drawNormalPoints(numSamples,mu,sigma):
    from mc_utilities import drawStdNormalPoints

    StdNormalPoints = drawStdNormalPoints(numSamples)

    return mu + sigma*StdNormalPoints

def fnc2integrate(x,fncExpnt):
    if x < 0:
        fncValue = 0
    else:
        fncValue = x**fncExpnt 
    return fncValue

def calcAnalyticIntegral(sigma,fncExpnt):
    from scipy.special import gamma as gamma_fnc
    from math import sqrt
    from math import pi
    analyticIntegral = (1/(sqrt(2*pi)*sigma)) \
                        *0.5*gamma_fnc(0.5*(fncExpnt+1)) \
                        *(2*sigma**2)**(0.5*(fncExpnt+1))
#    pdb.set_trace()
    return analyticIntegral

def autoconversionRate(chi,Nc,alpha,beta):
    if chi < 0:
        fncValue = 0
    else:
        fncValue = chi**alpha * Nc**beta 
    return fncValue

def calcAnalyticIntegral2D(muChi,sigmaChi,muNcn,sigmaNcn,rChiNcn,alpha,beta):
    from scipy.special import gamma 
    from math import sqrt, exp, pi, pbdv

    sC = muChi/sigmaChi + rChiNcn*sigmaNcn*beta

    analyticIntegral = (1/sqrt(2*pi))*(sigmaChi**alpha) \
                        *exp(muNcn*beta + 0.5*(sigmaNcn*beta)**2 - 0.25*sC**2) \
                        *gamma(alpha+1)*pbdv(-alpha-1,-sC)

#    pdb.set_trace()
    return analyticIntegral
        
def computeRmseN(numSamples):
    from numpy import zeros, arange
    from mc_utilities import computeRmse, calcFncValues, integrateFncValues

#    print("In computeRmseN")
    fncDim = 1  # Dimension of uni- or multi-variate integrand function
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
    
#        fncValuesArray = calcFncValues(numSamples,normalPoints,fncExpnt)    
        fncValuesArray = calcFncValues(numSamples,fncDim,normalPoints,fnc2integrate,fncExpnt)
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

    