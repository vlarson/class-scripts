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

    from numpy import exp, multiply, sqrt, log, pi

    xN = samplePoints[:,0]     # Column of normally distributed sample points
    xLN = samplePoints[:,1]    # Column of lognormally distributed sample points, 
                               #    left in lognormal space

    prefactor = 1.0 / ( 2.0 * pi * sigmaN * sigmaLNn \
                        * sqrt( 1.0 - rhon**2 ) * xLN \
                      ) 

    exponent = - ( 1.0 / ( 2.0 * ( 1.0 - rhon**2 ) ) ) \
           * (   ( 1.0 / sigmaN**2 ) * ( xN - muN )**2 \
               - ( 2.0 * rhon / ( sigmaN * sigmaLNn ) ) \
                  * ( xN - muN ) * ( log( xLN ) - muLNn ) \
               + ( 1.0 / sigmaLNn**2 ) * ( log( xLN ) - muLNn )**2 \
              )
              
    PDFValues =  multiply( prefactor, exp( exponent ) )

    return PDFValues

def calcWeightsArrayImp(samplePointsQ,alphaDefQ,
                        muN,sigmaN,muLNn,sigmaLNn,rhon,
                        muNQ,sigmaNQ,muLNnQ,sigmaLNnQ):
    """Given a sample, return importance weights P(x)/q(x).
    
    Inputs:
    samplePointsQ = a 2D array of sample points of size numSamples. 
                    Column 0 contains normally distributed points.  
                    Column 1 contains lognormally distributed points.  
                    First numSamplesDefP points in array come from P(x).
    alphaDefQ = Fraction of samples drawn from q(x) rather than P(x)
    numSamplesDefP = Number of samples drawn from P(x) rather than q(x)
    muN = mean of normal variate
    sigmaN = standard deviation of normal variate
    muLNn = mean of lognormal variate, transformed to normal space
    sigmaLNn = standard deviation of lognormal variate, transformed to normal space
    rhon = correlation between the 2 variates, transformed to normal space
    muNDeltaImp = importance muN - muN
    sigmaNDeltaImp = importance sigmaN - sigmaN
    muLNnDeltaImp = importance muLNn - muLNn 
    sigmaLNnDeltaImp = importance sigmaLNn - sigmaLNn    """

    from numpy import divide

    POfX = calcNormalLognormalPDFValues(samplePointsQ,
                                        muN,sigmaN,muLNn,sigmaLNn,
                                        rhon)

    qOfX = calcNormalLognormalPDFValues(samplePointsQ,
                                        muNQ,sigmaNQ,muLNnQ,sigmaLNnQ,
                                        rhon)

    qOfXAlpha = alphaDefQ * qOfX + (1-alphaDefQ) * POfX

    weightsArrayImp = divide( POfX, qOfXAlpha ) 

    #pdb.set_trace()

    return weightsArrayImp

        
def computeFracRmseN(numSamples):
    """Return the fractional root-mean-square error 
    in a Monte-Carlo integration of Khairoutdinov-Kogan autoconversion.
    
    As we go, optionally produce plots."""    
    
    
    from numpy import zeros, arange, copy, cov, corrcoef, any, nan, \
                            clip, finfo, amax, multiply, mean, divide, power, \
                            floor, concatenate
    from mc_utilities import computeRmse, calcFncValues, integrateFncValues
    from math import isnan
    import matplotlib.pyplot as plt
    import sys

#    print("In computeRmseN")
    fncDim = 2  # Dimension of uni- or multi-variate integrand function
    muChi = 0
    sigmaChi = 1
    muNcn = 0
    sigmaNcn = 0.5
    rhoChiNcn = 0.5
    alpha = 2.47 #2.47
    beta = -1.79 #-1.79
    # Control variate parameters
    alphaDelta = -0.3  # Increment to alpha for control variates function, h
    betaDelta = -0.3  # Increment to beta for control variates function, h

    # Importance sampling parameters: Importance values - Basic MC values
    muChiDeltaImp = 1.8 * sigmaChi  # 1.4 * sigmaChi
    sigmaChiDeltaImp = -0.00 * sigmaChi
    muNcnDeltaImp = -1.0 * sigmaNcn
    sigmaNcnDeltaImp = -0.00 * sigmaNcn
    
    # Defensive sampling parameter
    alphaDefQ = 0.5  # Fraction of points drawn from q(x) rather than P(x)

    numExperiments = 100#1000
    
    createCVScatterplots = False

    mcIntegral = zeros(numExperiments)
    mcIntegralImp = zeros(numExperiments)
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

    muChiQ = muChi + muChiDeltaImp
    sigmaChiQ = sigmaChi + sigmaChiDeltaImp
    muNcnQ = muNcn + muNcnDeltaImp
    sigmaNcnQ = sigmaNcn+sigmaNcnDeltaImp

#    pdb.set_trace()

    for idx in arange(numExperiments):

        #pdb.set_trace()

        samplePoints = drawNormalLognormalPoints( numSamples,
                                                  muChi,sigmaChi,
                                                  muNcn,sigmaNcn,
                                                  rhoChiNcn)

#        pdb.set_trace()    
    
        fncValuesArray = calcFncValues(numSamples,fncDim,samplePoints,
                                       autoconversionRate,alpha,beta)
        #    print"Function values = %s" % fncValuesArray

        mcIntegral[idx] = integrateFncValues(fncValuesArray,numSamples)
        #print "Monte Carlo estimate = %s" % mcIntegral[idx]

        #########################################
        #
        # Calculate integral using control variates
        #
        ############################################

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
           
        mcIntegralCV[idx] = integrateFncValues(fncValuesArray-betaOpt*fncValuesArrayCV,numSamples) \
                            + betaOpt*analyticIntegralCV
        #print "CV Monte Carlo estimate = %s" % mcIntegralCV[idx] 
    
        #pdb.set_trace()    
        if isnan(mcIntegralCV[idx]):
            pdb.set_trace()


        #########################################
        #
        # Calculate integral using importance sampling (+ control variate)
        #
        ############################################

        # Number of samples drawn from q(x) ( and not P(x) ) in defensive importance sampling
        numSamplesDefQ = floor( alphaDefQ * numSamples ).astype(int) 

        # Number of samples drawn from q(x) ( and not P(x) ) in defensive importance sampling
        numSamplesDefP = numSamples-numSamplesDefQ 
        
        # Draw numSamplesDefQ samples from q(x), without including defensive points from P(x)
        samplePointsQOnly = drawNormalLognormalPoints( numSamplesDefQ,
                                                  muChiQ,sigmaChiQ,muNcnQ,sigmaNcnQ,
                                                  rhoChiNcn)

        # Concatenate sample points drawn from q(x) and P(x) 
        # P(x) points come first
        samplePointsQ = concatenate( ( samplePoints[0:numSamplesDefP,:], samplePointsQOnly ),
                                    axis=0 )   # Add rows to the bottom of the 2-column array                                     

        #pdb.set_trace()
                                    
        # Assertion check:
        if ( samplePointsQ.shape != samplePoints.shape  ):
            print("ERROR: Defensive importance sampling generates the wrong number of sample points!!!!") 
            sys.exit(1)        

        fncValuesArrayQ = calcFncValues(numSamples,fncDim,samplePointsQ,
                                       autoconversionRate,alpha,beta)

        fncValuesArrayCVQ = calcFncValues(numSamples,fncDim,samplePointsQ,
                                       autoconversionRate,alpha+alphaDelta,beta+betaDelta)                                       

        weightsArrayImp = calcWeightsArrayImp(samplePointsQ,alphaDefQ,
                        muChi,sigmaChi,muNcn,sigmaNcn,rhoChiNcn,
                        muChiQ,sigmaChiQ,muNcnQ,sigmaNcnQ)

        # Effective sample size
        neOnN = divide( (mean(weightsArrayImp))**2,
                        mean(power(weightsArrayImp,2)) 
                      )
        #print "Effective sample size = neOnN = %s" % neOnN
 
        betaCVQ = 0.0
        #betaCVQ = betaOpt # betaOpt is the optimal beta for non-importance sampling

        integrandArrayImp = multiply( fncValuesArrayQ-betaCVQ*fncValuesArrayCVQ, 
                                     weightsArrayImp  )

        mcIntegralImp[idx] = integrateFncValues(integrandArrayImp,numSamples) \
                            + betaCVQ*analyticIntegralCV

        #pdb.set_trace()
   
    fracRmse = computeRmse(analyticIntegral,mcIntegral)/analyticIntegral
    print("Fractional RMSE of Monte Carlo estimate = %s" % fracRmse)

#    pdb.set_trace()
    
    fracRmseImp = computeRmse(analyticIntegral,mcIntegralImp)/analyticIntegral
    print("Fractional RMSE of Monte Carlo estimate = %s" % fracRmse)    
    
    fracRmseCV = computeRmse(analyticIntegral,mcIntegralCV)/analyticIntegral
    print("Fractional RMSE of CV Monte Carlo estimate = %s" % fracRmseCV)    

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
    
    return (fracRmse, fracRmseImp, fracRmseCV, corrCV[0,1])    
    
def main():
    from numpy import zeros, arange, sqrt, divide, abs
    import matplotlib.pyplot as plt
    
    numNValues = 10#20#10 # Number of trials with different sample size

    fracRmseNValues = zeros(numNValues) 
    fracRmseNValuesImp = zeros(numNValues)
    fracRmseNValuesCV = zeros(numNValues)
    corrCVNValues = zeros(numNValues)
    numSamplesN = zeros(numNValues).astype(int)
    
    for idx in arange(numNValues):    
        numSamplesN[idx] =  2**(idx+2)
        print("numSamplesN = %s" % numSamplesN[idx])
        fracRmseNValues[idx], fracRmseNValuesImp[idx], fracRmseNValuesCV[idx], \
        corrCVNValues[idx] =  \
            computeFracRmseN(numSamplesN[idx])
    
    theoryError = 10.0/sqrt(numSamplesN)    

#    pdb.set_trace()
    plt.ion() # Use interactive mode so that program continues when plot appears    
    plt.clf()
#    plt.subplot(221)
    plt.loglog(numSamplesN, fracRmseNValues, label='Fractional MC Error') 
    plt.loglog(numSamplesN, fracRmseNValuesImp, label='Fractional Imp MC Error')
    plt.loglog(numSamplesN, fracRmseNValuesCV, label='Fractional CV MC Error')
    plt.loglog(numSamplesN, theoryError, label='Theory (1/sqrt(N))')
    plt.legend()
    plt.xlabel('Number of sample points')
    plt.ylabel('Root-mean-square error')
    plt.figure()

    plt.clf()
    plt.semilogx( numSamplesN, divide( fracRmseNValuesCV, fracRmseNValues ), label="Sample CV Err" )
    plt.semilogx( numSamplesN, sqrt( abs( 1 - corrCVNValues**2 ) ), label="sqrt(|1-rho**2|)" )
    plt.xlabel('Number of sample points')
    plt.ylabel('Data and theoretical estimate  [-]')
    plt.title('Control variate RMSE normalized by MC RMSE')
    plt.legend()


    plt.show()    
    
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()

    