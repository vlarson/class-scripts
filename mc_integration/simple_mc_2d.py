# -*- coding: utf-8 -*-
"""  A function that performs simple Monte Carlo integration """

import pdb

# pdb.set_trace()

def autoconversionRate(TwoDSamplePoint,alpha,beta):
    """Return a quantity proportional to the Khairoutdinov-Kogan autoconversion rate."""
    
    chi = TwoDSamplePoint[0]
    Nc  = TwoDSamplePoint[1]
    if chi < 0:
        fncValue = 0
    else:
        fncValue = chi**alpha * Nc**beta 

#    pdb.set_trace()
    
    return fncValue

def evaporationRate(TwoDSamplePoint,alpha,beta):
    """A simple function that mimics an evaporation formula."""
    
    chi = TwoDSamplePoint[0]
    Nc  = TwoDSamplePoint[1]
    if chi > 0:
        fncValue = 0
    else:
        fncValue = abs(chi)**alpha * Nc**beta 

#    pdb.set_trace()
    
    return fncValue

def calcAutoconversionIntegral(muChi,sigmaChi,muNcn,sigmaNcn,rChiNcn,alpha,beta):
    """Calculate the Khairoutdinov-Kogan autoconversion rate, 
    upscaled over a single normal-lognormal PDF.""" 

    from scipy.special import gamma, pbdv 
    from math import sqrt, exp, pi

    sC = muChi/sigmaChi + rChiNcn*sigmaNcn*beta
#    sC = muChi/sigmaChi - rChiNcn*sigmaNcn*beta

    (parabCylFnc, parabCylDer) = pbdv(-alpha-1,-sC)
#    (parabCylFnc, parabCylDer) = pbdv(-alpha-1,sC)


    analyticIntegral = (1/sqrt(2*pi))*(sigmaChi**alpha) \
                        *exp(muNcn*beta + 0.5*(sigmaNcn*beta)**2 - 0.25*sC**2) \
                        *gamma(alpha+1)*parabCylFnc

#    pdb.set_trace()

    return analyticIntegral

def drawNormalLognormalPoints(numSamples,muN,sigmaN,muLNn,sigmaLNn,rn):
    """Return sample points from a non-standard normal-lognormal PDF."""

    from mc_utilities import drawStdNormalPoints
    from numpy import zeros, exp, dot, copy
    from numpy.linalg import cholesky

    stdNormalPoints = zeros((numSamples,2))
    
    stdNormalPoints[:,0] = drawStdNormalPoints(numSamples)
    stdNormalPoints[:,1] = drawStdNormalPoints(numSamples)

    covarMatn = [  [sigmaN**2,            rn*sigmaN*sigmaLNn],
                   [rn*sigmaN*sigmaLNn,   sigmaLNn**2]
                ]

    LCholesky = cholesky(covarMatn)

#    normalPoints = dot(stdNormalPoints, LCholesky) + [muN, muLNn]
    normalPoints = dot(stdNormalPoints, LCholesky.T) + [muN, muLNn]

#    pdb.set_trace()

    normalLognormalPoints = copy(normalPoints)
    normalLognormalPoints[:,1] = exp(normalLognormalPoints[:,1])

    return normalLognormalPoints

        
def computeFracRmseN(numSamples):
    """Return the fractional root-mean-square error 
    in a Monte-Carlo integration of Khairoutdinov-Kogan autoconversion."""    
    
    
    from numpy import zeros, arange, copy, cov, any, nan, clip, finfo, amax
    from mc_utilities import computeRmse, calcFncValues, integrateFncValues
    from math import isnan
    import matplotlib.pyplot as plt 

#    print("In computeRmseN")
    fncDim = 2  # Dimension of uni- or multi-variate integrand function
    muChi = 0
    sigmaChi = 1
    muNcn = 0
    sigmaNcn = 1.5
    rChiNcn = 0.5
    alpha = 2.47 #2.47
    beta = -1.79 #-1.79
    # Control variate parameters
    alphaDelta = -0.3  # Increment to alpha for control variates function, h
    betaDelta = -0.3  # Increment to beta for control variates function, h

    numExperiments = 100#1000

    mcIntegral = zeros(numExperiments)
    mcIntegralEvap = zeros(numExperiments)
    mcIntegralCV = zeros(numExperiments)

    analyticIntegral =  calcAutoconversionIntegral( muChi,sigmaChi,
                                                    muNcn,sigmaNcn,
                                                    rChiNcn,
                                                    alpha,beta
                                                  )
    #print "Analytic calculation of true integral = %s" % analyticIntegral

    analyticIntegralCV =  calcAutoconversionIntegral( muChi,sigmaChi,
                                                    muNcn,sigmaNcn,
                                                    rChiNcn,
                                                    alpha+alphaDelta,beta+betaDelta
                                                  )
    #print "Analytic calculation of CV integral = %s" % analyticIntegralCV

#    pdb.set_trace()


    for idx in arange(numExperiments):

        #pdb.set_trace()

        samplePoints = drawNormalLognormalPoints( numSamples,
                                                  muChi,sigmaChi,
                                                  muNcn,sigmaNcn,
                                                  rChiNcn)
        
#        samplePointsEvap = copy(samplePoints)
#        # Reverse sign of s points
#        samplePointsEvap[:,0] = -1.0*samplePointsEvap[:,0]
        samplePointsEvap = drawNormalLognormalPoints( numSamples,
                                                  -muChi,sigmaChi,
                                                  muNcn,sigmaNcn,
                                                  -rChiNcn)

#        pdb.set_trace()    
#        fncValuesArray = calcFncValues(numSamples,normalPoints,fncExpnt)    
        fncValuesArray = calcFncValues(numSamples,fncDim,samplePoints,
                                       autoconversionRate,alpha,beta)

        fncValuesArrayEvap = calcFncValues(numSamples,fncDim,samplePointsEvap,
                                       evaporationRate,alpha,beta)
#    print"Function values = %s" % fncValuesArray  
        fncValuesArrayCV = calcFncValues(numSamples,fncDim,samplePoints,
                                       autoconversionRate,alpha+alphaDelta,beta+betaDelta)                                       

        if any(fncValuesArrayCV==nan):
            pdb.set_trace()



        #pdb.set_trace()
        # Compute optimal beta for control variates
        covCV = cov(fncValuesArray,fncValuesArrayCV)

        betaOpt = covCV[0,1]/amax([ covCV[1,1] , finfo(float).eps ])
        betaOpt = clip(betaOpt, 0.0, 1.0)
        #print "betaOpt = %s" % betaOpt
        #print "covCV = %s" % covCV
    
        mcIntegral[idx] = integrateFncValues(fncValuesArray,numSamples)
        #print "Monte Carlo estimate = %s" % mcIntegral[idx]

        mcIntegralEvap[idx] = integrateFncValues(fncValuesArrayEvap,numSamples)
        #print "Evaporation Monte Carlo estimate = %s" % mcIntegralEvap[idx]
        
        mcIntegralCV[idx] = integrateFncValues(fncValuesArray-betaOpt*fncValuesArrayCV,numSamples) \
                            + betaOpt*analyticIntegralCV
        #print "CV Monte Carlo estimate = %s" % mcIntegralCV[idx] 
    
        #pdb.set_trace()    
        if isnan(mcIntegralCV[idx]):
            pdb.set_trace()

    
    fracRmse = computeRmse(analyticIntegral,mcIntegral)/analyticIntegral
    print "Fractional RMSE of Monte Carlo estimate = %s" % fracRmse

#    pdb.set_trace()
    
    fracRmseCV = computeRmse(analyticIntegral,mcIntegralCV)/analyticIntegral
    print "Fractional RMSE of CV Monte Carlo estimate = %s" % fracRmseCV    

#    if isnan(fracRmseCV):
#    pdb.set_trace

    plt.scatter(fncValuesArray,fncValuesArrayCV)
    plt.plot([min(fncValuesArray), max(fncValuesArray)], 
             [min(fncValuesArray), max(fncValuesArray)])
    plt.grid()
    plt.xlabel('Original function values')
    plt.ylabel('Control variate function values')
    plt.show()

#    pdb.set_trace()
    
    return (fracRmse, fracRmseCV)    
    
def main():
    from numpy import zeros, arange, sqrt
    import matplotlib.pyplot as plt
    
    numNValues = 10#20#10 # Number of trials with different sample size

    fracRmseNValues = zeros(numNValues)    
    fracRmseNValuesCV = zeros(numNValues)
    numSamplesN = zeros(numNValues)
    
    for idx in arange(numNValues):    
        numSamplesN[idx] =  2**(idx+2)
        print "numSamplesN = %s" % numSamplesN[idx]
        fracRmseNValues[idx], fracRmseNValuesCV[idx] = computeFracRmseN(numSamplesN[idx])
    
    theoryError = 10.0/sqrt(numSamplesN)    

#    pdb.set_trace()
    
    plt.clf()
#    plt.subplot(221)
    plt.loglog(numSamplesN, fracRmseNValues, label='Fractional MC Error') 
    plt.loglog(numSamplesN, fracRmseNValuesCV, label='Fractional CV MC Error')
    plt.loglog(numSamplesN, theoryError, label='Theory (1/sqrt(N))')
    plt.legend()
    plt.xlabel('Number of sample points')
    plt.ylabel('Root-mean-square error')
    plt.show()    
    
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()

    