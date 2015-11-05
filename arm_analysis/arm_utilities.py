# -*- coding: utf-8 -*-
"""
   Analyze ARM observations.  
"""


def plotSfcRad(beflux_file, min_sdn):
    """Plot surface radiative fields."""

# Point to directory containing ARM observations 
#data_dir = '/home/studi/Larson/arm_data_files/'

#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131204.000000.custom.cdf'
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131205.000000.custom.cdf'
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131206.000000.custom.cdf'
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131207.000000.custom.cdf'
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131208.000000.custom.cdf'
# SDN showed a few clouds on 20131215:
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131215.000000.custom.cdf'
# 20131217 had essentially clear skies
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131217.000000.custom.cdf'
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131218.000000.custom.cdf'


    # Import libraries
    from numpy import fmax
    import matplotlib.pyplot as plt
    from scipy.io import netcdf
    import pdb

    beflux_nc = netcdf.netcdf_file(beflux_file, 'r')

    time_offset_beflux = beflux_nc.variables['time_offset']
    short_direct_normal = beflux_nc.variables['short_direct_normal']

    # Replace small values with threshold, for plotting time series
    sdn_floored = fmax(min_sdn,short_direct_normal[:])

    # Remove small values from time series, thereby shortening vector length
    sdn_truncated = [x for x in short_direct_normal if x > min_sdn]

    #pdb.set_trace()

    # Plot time series of shortwave direct normal flux
    plt.clf()
    plt.subplot(211)
    plt.plot(time_offset_beflux[:],sdn_floored[:])
    plt.xlabel('Time')
    plt.ylabel('Shortwave direct normal flux')

    #pdb.set_trace()

    # Plot histogram of shortwave direct normal flux
    plt.subplot(212)
    n, bins, patches = plt.hist(sdn_truncated[:], 50, normed=1, histtype='stepfilled')
    plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    plt.xlabel('Shortwave direct normal flux')
    plt.ylabel('Probability')
    plt.figure()
    plt.show()

    return
        
def findTruncNormalRoots(truncMean,truncVarnce,muInit,sigmaInit,min_refl):
    """Returns parameters mu and sigma of a truncated normal distribution.
    
    Inputs:    
    truncMean = mean of truncated part of full (untruncated) normal
    truncVarnce = variance of truncated part of full normal
    muInit = first guess value of mu, the mean of the full normal    
    sigmaInit = first guess value of sigma, the standard deviation of the full normal
    min_refl = minimum (left) threshold value at which normal is truncated
    """
    
    from scipy.optimize import root  

    sol = root(findTruncMeanVarnceFncResid, [muInit, sigmaInit], 
               args=(truncMean,truncVarnce,min_refl), jac=False, method='hybr')

    mu = sol.x[0]
    sigma = sol.x[1]
    
    return (mu, sigma)
    
def findTruncMeanVarnceFncResid(x,truncMean,truncVarnce,min_refl):
    """Evaluates the residual of the function that we want to zero 
    in order to solve for the parameters of a truncated normal distribution.
    
    Inputs:
    x[0] = mean of untruncated (full) normal
    x[1] = standard deviation of untruncated normal
    truncMean = mean of truncated part of full normal
    truncVarnce = variance of truncated part of full normal
    min_refl = minimum (left) threshold value at which normal is truncated
    
    Output:
    Residual of the function to be zeroed
    """


    mu = x[0]
    sigma = x[1]    
    
    alpha = (min_refl-mu)/sigma 
    lambdaAlpha = lambdaFnc(alpha)
    
    truncMeanResid = mu + sigma * lambdaAlpha - truncMean 
    
    truncVarnceResid = sigma**2 * ( 1.0 - deltaFnc(alpha,lambdaAlpha) ) - truncVarnce    
    
    return (truncMeanResid, truncVarnceResid)
    
def lambdaFnc(alpha):
    """A utility function used to relate moments and parameters of a truncated normal."""
    
    from scipy.stats import norm
    from numpy import finfo, amax        
        
        
    return norm.pdf(alpha) / amax([ 1.0 - norm.cdf(alpha) , finfo(float).eps ])

def deltaFnc(alpha,lambdaAlpha):
    """A utility function used to relate variance and sigma parameter of a truncated normal."""
    
    return lambdaAlpha * ( lambdaAlpha - alpha )