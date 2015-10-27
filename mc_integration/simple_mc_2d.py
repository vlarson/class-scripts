# -*- coding: utf-8 -*-
"""  A function that performs simple Monte Carlo integration """

import pdb

# pdb.set_trace()

def autoconversionRate(TwoDSamplePoint,alpha,beta):
    
    chi = TwoDSamplePoint[0]
    Nc  = TwoDSamplePoint[1]
    if chi < 0:
        fncValue = 0
    else:
        fncValue = chi**alpha * Nc**beta 

#    pdb.set_trace()
    
    return fncValue

def calcAutoconversionIntegral(muChi,sigmaChi,muNcn,sigmaNcn,rChiNcn,alpha,beta):
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
    from mc_utilities import drawStdNormalPoints
    from numpy import zeros, exp, dot
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

    normalLognormalPoints = normalPoints
    normalLognormalPoints[:,1] = exp(normalLognormalPoints[:,1])

    return normalLognormalPoints

        
def computeFracRmseN(numSamples):
    from numpy import zeros, arange
    from mc_utilities import computeRmse, calcFncValues, integrateFncValues

#    print("In computeRmseN")
    fncDim = 2  # Dimension of uni- or multi-variate integrand function
    muChi = 0
    sigmaChi = 1
    muNcn = 0
    sigmaNcn = 1.5
    rChiNcn = -0.0
    alpha = 0 #2.47
    beta = -1 #-1.79

    numExperiments = 1

    mcIntegral = zeros(numExperiments)

    analyticIntegral =  calcAutoconversionIntegral( muChi,sigmaChi,
                                                    muNcn,sigmaNcn,
                                                    rChiNcn,
                                                    alpha,beta
                                                  )
#    pdb.set_trace()

    print "Analytic calculation of integral = %s" % analyticIntegral

    for idx in arange(numExperiments):

        samplePoints = drawNormalLognormalPoints( numSamples,
                                                  muChi,sigmaChi,
                                                  muNcn,sigmaNcn,
                                                  rChiNcn)
#    print"NormalPoints = %s" % normalPoints

#        pdb.set_trace()    
#        fncValuesArray = calcFncValues(numSamples,normalPoints,fncExpnt)    
        fncValuesArray = calcFncValues(numSamples,fncDim,samplePoints,
                                       autoconversionRate,alpha,beta)
#    print"Function values = %s" % fncValuesArray  

#        pdb.set_trace()
    
        mcIntegral[idx] = integrateFncValues(fncValuesArray,numSamples)
        print "Monte Carlo estimate = %s" % mcIntegral[idx]    
        
    
    fracRmse = computeRmse(analyticIntegral,mcIntegral)/analyticIntegral
    print "Fractional RMSE of Monte Carlo estimate = %s" % fracRmse
    
    return fracRmse    
    
def main():
    from numpy import zeros, arange, sqrt
    import matplotlib.pyplot as plt
    
    numNValues = 20#10 # Number of trials with different sample size

    fracRmseNValues = zeros(numNValues)    
    numSamplesN = zeros(numNValues)
    
    for idx in arange(numNValues):    
        numSamplesN[idx] = 2**(idx+2)
        print "numSamplesN = %s" % numSamplesN[idx]
        fracRmseNValues[idx] = computeFracRmseN(numSamplesN[idx])
    
    theoryError = 10.0/sqrt(numSamplesN)    
    
    plt.clf()
#    plt.subplot(221)
    plt.loglog(numSamplesN, fracRmseNValues, label='Fractional MC Error')    
    plt.loglog(numSamplesN, theoryError, label='Theory (1/sqrt(N))')
    plt.legend()
    plt.xlabel('Number of sample points')
    plt.ylabel('Root-mean-square error')
    plt.show()    
    
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()

    