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

def calcAutoconversionIntegral(muChi,sigmaChi,muNcn,sigmaNcn,rhoChiNcn,alpha,beta):
    """Calculate the Khairoutdinov-Kogan autoconversion rate, 
    upscaled over a single normal-lognormal PDF.""" 

    from scipy.special import gamma, pbdv 
    from math import sqrt, exp, pi

    sC = muChi/sigmaChi + rhoChiNcn*sigmaNcn*beta
#    sC = muChi/sigmaChi - rhoChiNcn*sigmaNcn*beta

    (parabCylFnc, parabCylDer) = pbdv(-alpha-1,-sC)
#    (parabCylFnc, parabCylDer) = pbdv(-alpha-1,sC)


    analyticIntegral = (1/sqrt(2*pi))*(sigmaChi**alpha) \
                        *exp(muNcn*beta + 0.5*(sigmaNcn*beta)**2 - 0.25*sC**2) \
                        *gamma(alpha+1)*parabCylFnc

#    pdb.set_trace()

    return analyticIntegral

def drawNormalLognormalPoints(numSamples,muN,sigmaN,muLNn,sigmaLNn,rhon):
    """Return sample points from a non-standard normal-lognormal PDF."""

    from mc_utilities import drawStdNormalPoints
    from numpy import zeros, exp, dot, copy
    from numpy.linalg import cholesky

    stdNormalPoints = zeros((numSamples,2))
    
    stdNormalPoints[:,0] = drawStdNormalPoints(numSamples)
    stdNormalPoints[:,1] = drawStdNormalPoints(numSamples)

    covarMatn = [  [sigmaN**2,            rhon*sigmaN*sigmaLNn],
                   [rhon*sigmaN*sigmaLNn,   sigmaLNn**2]
                ]

    LCholesky = cholesky(covarMatn)

#    normalPoints = dot(stdNormalPoints, LCholesky) + [muN, muLNn]
    normalPoints = dot(stdNormalPoints, LCholesky.T) + [muN, muLNn]

#    pdb.set_trace()

    normalLognormalPoints = copy(normalPoints)
    normalLognormalPoints[:,1] = exp(normalLognormalPoints[:,1])

    return normalLognormalPoints

def calcNormalLognormalPDFValues(samplePoints,muN,sigmaN,muLNn,sigmaLNn,rhon):
    """Given a sample, return values of a normal-lognormal PDF.
    
    Inputs:
    samplePoints = a 2D array of sample points. Column 0 contains normally
                    distributed points.  Column 1 contains lognormally 
                    distributed points.
    muN = mean of normal variate
    sigmaN = standard deviation of normal variate
    muLNn = mean of lognormal variate, transformed to normal space
    sigmaLNn = standard deviation of lognormal variate, transformed to normal space
    rhon = correlation between the 2 variates, transformed to normal space
    """

    from numpy import zeros, exp, dot, copy, multiply, sqrt, log, pi

    xN = samplePoints[:,0]     # Column of normally distributed sample points
    xLN = samplePoints[:,1]    # Column of lognormally distributed sample points

    prefactor = 1.0 / ( 2.0 * pi * sigmaN * sigmaLNn \
          * sqrt( 1.0 - rhon**2 ) * xLN ) 

    exponent = - ( 1.0 / ( 2.0 * ( 1.0 - rhon**2 ) ) ) \
           * (   ( 1.0 / sigmaN**2 ) * ( xN - muN )**2 \
               - ( 2.0 * rhon / ( sigmaN * sigmaLNn ) ) \
                  * ( xN - muN ) * ( log( xLN ) - muLNn ) \
               + ( 1.0 / sigmaLNn**2 ) * ( log( xLN ) - muLNn )**2 \
              )
              
    PDFValues =  multiply( prefactor, exponent )

    return PDFValues
        
def computeFracRmseN(numSamples):
    """Return the fractional root-mean-square error 
    in a Monte-Carlo integration of Khairoutdinov-Kogan autoconversion.
    
    As we go, optionally produce plots."""    
    
    
    from numpy import zeros, arange, copy, cov, corrcoef, any, nan, clip, finfo, amax
    from mc_utilities import computeRmse, calcFncValues, integrateFncValues
    from math import isnan
    import matplotlib.pyplot as plt 

#    print("In computeRmseN")
    fncDim = 2  # Dimension of uni- or multi-variate integrand function
    muChi = 0
    sigmaChi = 1
    muNcn = 0
    sigmaNcn = 1.5
    rhoChiNcn = 0.5
    alpha = 2.47 #2.47
    beta = -1.79 #-1.79
    # Control variate parameters
    alphaDelta = -0.3  # Increment to alpha for control variates function, h
    betaDelta = -0.3  # Increment to beta for control variates function, h

    numExperiments = 100#1000
    
    createCVScatterplots = False

    mcIntegral = zeros(numExperiments)
    mcIntegralEvap = zeros(numExperiments)
    mcIntegralCV = zeros(numExperiments)

    analyticIntegral =  calcAutoconversionIntegral( muChi,sigmaChi,
                                                    muNcn,sigmaNcn,
                                                    rhoChiNcn,
                                                    alpha,beta
                                                  )
    #print "Analytic calculation of true integral = %s" % analyticIntegral

    analyticIntegralCV =  calcAutoconversionIntegral( muChi,sigmaChi,
                                                    muNcn,sigmaNcn,
                                                    rhoChiNcn,
                                                    alpha+alphaDelta,beta+betaDelta
                                                  )
    #print "Analytic calculation of CV integral = %s" % analyticIntegralCV

#    pdb.set_trace()


    for idx in arange(numExperiments):

        #pdb.set_trace()

        samplePoints = drawNormalLognormalPoints( numSamples,
                                                  muChi,sigmaChi,
                                                  muNcn,sigmaNcn,
                                                  rhoChiNcn)

#        pdb.set_trace()    
#        fncValuesArray = calcFncValues(numSamples,normalPoints,fncExpnt)    
        fncValuesArray = calcFncValues(numSamples,fncDim,samplePoints,
                                       autoconversionRate,alpha,beta)

#    print"Function values = %s" % fncValuesArray  
        fncValuesArrayCV = calcFncValues(numSamples,fncDim,samplePoints,
                                       autoconversionRate,alpha+alphaDelta,beta+betaDelta)                                       

        if any(fncValuesArrayCV==nan):
            pdb.set_trace()

        #pdb.set_trace()
        # Compute optimal beta (pre-factor for control variate)
        covCV = cov(fncValuesArray,fncValuesArrayCV)
        #print "covCV = %s" % covCV

        # Optimal beta
        betaOpt = covCV[0,1]/amax([ covCV[1,1] , finfo(float).eps ])
        #betaOpt = clip(betaOpt, 0.0, 1.0)
        #print "betaOpt = %s" % betaOpt

        corrCV = corrcoef(fncValuesArray,fncValuesArrayCV)

        # pdb.set_trace()
    
        mcIntegral[idx] = integrateFncValues(fncValuesArray,numSamples)
        #print "Monte Carlo estimate = %s" % mcIntegral[idx]
        
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
    if ( createCVScatterplots == True ):
        plt.scatter(fncValuesArray,fncValuesArrayCV)
        plt.plot([min(fncValuesArray), max(fncValuesArray)], 
                 [min(fncValuesArray), max(fncValuesArray)])
        plt.grid()
        plt.xlabel('Original function values')
        plt.ylabel('Control variate function values')
        plt.show()

#    pdb.set_trace()
    
    return (fracRmse, fracRmseCV, corrCV[0,1])    
    
def main():
    from numpy import zeros, arange, sqrt, divide
    import matplotlib.pyplot as plt
    
    numNValues = 10#20#10 # Number of trials with different sample size

    fracRmseNValues = zeros(numNValues)    
    fracRmseNValuesCV = zeros(numNValues)
    corrCVNValues = zeros(numNValues)
    numSamplesN = zeros(numNValues)
    
    for idx in arange(numNValues):    
        numSamplesN[idx] =  2**(idx+2)
        print "numSamplesN = %s" % numSamplesN[idx]
        fracRmseNValues[idx], fracRmseNValuesCV[idx], corrCVNValues[idx] =  computeFracRmseN(numSamplesN[idx])
    
    theoryError = 10.0/sqrt(numSamplesN)    

#    pdb.set_trace()
    plt.ion() # Use interactive mode so that program continues when plot appears    
    plt.clf()
#    plt.subplot(221)
    plt.loglog(numSamplesN, fracRmseNValues, label='Fractional MC Error') 
    plt.loglog(numSamplesN, fracRmseNValuesCV, label='Fractional CV MC Error')
    plt.loglog(numSamplesN, theoryError, label='Theory (1/sqrt(N))')
    plt.legend()
    plt.xlabel('Number of sample points')
    plt.ylabel('Root-mean-square error')
    plt.figure()

    plt.clf()
    plt.semilogx( numSamplesN, divide( fracRmseNValuesCV, fracRmseNValues ), label="Sample CV Err" )
    plt.semilogx( numSamplesN, sqrt( 1 - corrCVNValues**2 ), label="sqrt(1-rho**2)" )
    plt.xlabel('Number of sample points')
    plt.ylabel('Data and theoretical estimate  [-]')
    plt.title('Control variate RMSE normalized by MC RMSE')
    plt.legend()


    plt.show()    
    
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()

    