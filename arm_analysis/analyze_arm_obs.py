# -*- coding: utf-8 -*-
"""
   Analyze ARM observations.  
"""

# Import libraries
from __future__ import division # in order to divide two integers
from numpy import fmax, arange, meshgrid, ix_, sqrt, mean, var, std, sum 
from numpy import linspace, asarray, sort, amin, zeros, isclose, count_nonzero
from numpy.ma import masked_where, filled
from numpy.ma import MaskedArray
from math import pi, log
from scipy.stats import norm, lognorm, skew, spearmanr
from scipy.stats.mstats import rankdata
import matplotlib.pyplot as plt
from scipy.io import netcdf
from arm_utilities import plotSfcRad, findTruncNormalRoots, findKSDn, calcMeanAlbedo
import pdb
import sys


# Point to directory containing ARM observations 
data_dir = '/home/studi/Larson/arm_data_files/'

##################################
#
#  Plot surface radiative fields
#
##################################

#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131204.000000.custom.cdf'
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131205.000000.custom.cdf'
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131206.000000.custom.cdf'
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131207.000000.custom.cdf'
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131208.000000.custom.cdf'
# SDN showed a few clouds on 20131215:
beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131215.000000.custom.cdf'
# 20131217 had essentially clear skies
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131217.000000.custom.cdf'
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131218.000000.custom.cdf'

# Impose a threshold on short_direct_normal to get rid of nighttime values
min_sdn = 10

#plotSfcRad(beflux_file, min_sdn)


##################################
#
#  Plot radar obs
#
##################################

# Default values
# Grid level at which to plot time series and histogram
range_level = 117
# Time at which profile of reflectivity is plotted
time_of_cloud = 69000 
# Number of profiles of reflectivity to be plotted
numProfiles = 5
# Indices for range of altitudes for time-height plots
height_range = arange(0,200)
# Indices for range of times for time-height plots
beginTimeOfPlot = 1
endTimeOfPlot = 86399
# Range of times in seconds for vertical overlap analysis
beginTimeOfCloud = 70000
endTimeOfCloud = 80000
# Range of altitudes in meters for vertical overlap analysis
cloudBaseHeight = 2000
cloudTopHeight = 3000
# If lArscl = True, then we want to use the ARSCL radar retrieval
radarType = "arscl"
# Impose a threshold on reflectivity_copol to get rid of noise values
if (radarType == "arscl"):
    minThreshRefl = -60
else:
    minThreshRefl = -30
# An estimate of within-cloud liquid water path, in g/m**2
meanLWP = 10 
    
    
# Now overwrite defaults with specialized values for particular days

#date = 20131204    # Shallow Sc
#date = 20131205    # Deep Cu
#date = 20131206    # Shallow Sc, bad data?
#date = 20131207    # Sc/Cu from 3 to 4 km
#date = 20131208    # Low drizzling Cu
#date = 20131215    # No clouds
#date = 20131217    # Noise
date = 20150607     # Shallow Cu and some mid level clouds
#date = 20150609     # Shallow Cu
#date = 20150627     # Shallow Cu

if date == 20131204:
    # Radar showed low stratus on 20131204:
    radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131204.000000.custom.nc'
    # Indices for range of altitudes for time-height plots
    height_range = arange(0,100)
    # Grid level at which to plot time series and histogram
    range_level = 18
elif date == 20131205:
    # Radar could see strong clouds up to 8 km on 20131205:
    radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131205.000002.custom.nc'
    # Grid level at which to plot time series and histogram    
    range_level = 167  
    # Indices for range of altitudes for time-height plots
    height_range = arange(50,250)
elif date == 20131206:
    # Radar could see clouds up to 2 km on 20131206:
    radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131206.000000.custom.nc'    
    # Grid level at which to plot time series and histogram
    range_level = 45
    # Time and time step at which profile of reflectivity is plotted
    time_of_cloud = 75000 
    # Indices for range of altitudes for time-height plots
    height_range = arange(1,100)
elif date == 20131207:
    # Radar could see Sc/Cu clouds from 3 km to 4 km on 20131207:
    radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131207.000001.custom.nc'    
    # Grid level at which to plot time series and histogram
    range_level = 110
elif date == 20131208:
    # Radar saw low drizzling cumulus on 20131208:
    radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131208.000003.custom.nc'
    # Indices for range of altitudes for time-height plots
    height_range = arange(0,35)
    # Grid level at which to plot time series and histogram
    range_level = 9
    # Impose a threshold on reflectivity_copol to get rid of nighttime values
    minThreshRefl = -40
elif date == 20131215:
    # Radar couldn't see clouds on 20131215:
    radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131215.000003.custom.nc'    
elif date == 20131217:
    # 20131217 had essentially clear skies
    radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131217.000003.custom.nc'    
elif date == 20150607:
    radarType = "arscl"
    # There should have been lots of clouds, but ARSCL could see few
    if ( radarType == "arscl" ): 
        radar_refl_file = data_dir + 'sgparsclkazr1kolliasC1.c1.20150607.000000.nc'
    elif ( radarType == "kazrCorge" ): 
        radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20150607.000000.nc'
    elif ( radarType == "kazrCormd" ):
        radar_refl_file = data_dir + 'sgpkazrcormdC1.c1.20150607.000000.nc'
    # Grid level at which to plot time series and histogram    
    range_level = 75 #80 
    # Indices for range of altitudes for time-height plots
    height_range = arange(0,100)
    beginTimeOfPlot = 68000
    endTimeOfPlot = 80000
    # Time at which profile of reflectivity is plotted
    time_of_cloud = 71000 #78410 #78800 #78450
    # Range of times in seconds for vertical overlap analysis
    beginTimeOfCloud = 70000 #70000#78000
    endTimeOfCloud = 79000 #72000#79000
    # Range of altitudes in meters for vertical overlap analysis
    cloudBaseHeight = 2250#2500
    cloudTopHeight = 2600#2800
    # An estimate of within-cloud liquid water path, in g/m**2
    meanLWP = 10
elif date == 20150609:
    radarType = "arscl"
    # Radar could see strong clouds up to 8 km on 20131205:
    if ( radarType == "arscl" ):
        radar_refl_file = data_dir + 'sgparsclkazr1kolliasC1.c1.20150609.000000.nc'
    elif ( radarType == "kazrCorge" ):
        radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20150609.000003.nc'
    elif ( radarType == "kazrCormd" ):
        radar_refl_file = data_dir + 'sgpkazrcormdC1.c1.20150609.000003.nc'                
    # Grid level at which to plot time series and histogram    
    range_level = 100  
    # Indices for range of altitudes for time-height plots
    height_range = arange(0,150)
    # Time and time step at which profile of reflectivity is plotted
    time_of_cloud = 66000 #76000
    # Range of times in seconds for vertical overlap analysis
    beginTimeOfCloud = 63000
    endTimeOfCloud = 78000
    # Range of altitudes in meters for vertical overlap analysis
    cloudBaseHeight = 2600
    cloudTopHeight = 3500
    # An estimate of within-cloud liquid water path, in g/m**2
    meanLWP = 8 
elif date == 20150627:
    radarType = "arscl"
    # Radar could see strong clouds up to 8 km on 20131205:
    if ( radarType == "arscl" ):
        radar_refl_file = data_dir + 'sgparsclkazr1kolliasC1.c1.20150627.000000.nc'
    elif ( radarType == "kazrCorge" ):
        radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20150627.000000.nc'
    elif ( radarType == "kazrCormd" ):
        radar_refl_file = data_dir + 'sgpkazrcormdC1.c1.20150627.000000.nc'
    # Grid level at which to plot time series and histogram    
    range_level = 95
    # Indices for range of altitudes for time-height plots
    height_range = arange(0,200)
    # Time and time step at which profile of reflectivity is plotted
    time_of_cloud = 67200 # At this time, the profile is highly correlated 
    time_of_cloud = 69400 # At this time, the profile is not well correlated
    time_of_cloud = 66300 #75000 #66000
    # Range of times in seconds for vertical overlap analysis
    beginTimeOfCloud = 63000
    endTimeOfCloud = 78000
    # Range of altitudes in meters for vertical overlap analysis
    cloudBaseHeight = 2200
    cloudTopHeight = 3500
    # An estimate of within-cloud liquid water path, in g/m**2
    meanLWP = 20 
else:
    print "Wrong date"


radar_refl_nc = netcdf.netcdf_file(radar_refl_file, 'r')

time_offset_radar_refl = radar_refl_nc.variables['time_offset']

# Compute beginning and ending time steps for time series and time-height plots
beginTimestepOfPlot = (abs(time_offset_radar_refl[:]-beginTimeOfPlot)).argmin()
endTimestepOfPlot = (abs(time_offset_radar_refl[:]-endTimeOfPlot)).argmin()
time_range = arange(beginTimestepOfPlot,endTimestepOfPlot)

# Compute time step for profile of snapshot
timestep_of_cloud = (abs(time_offset_radar_refl[:]-time_of_cloud)).argmin()
                    
# Compute beginning and ending time steps for block of cloud for overlap analysis
beginTimestepOfCloud = (abs(time_offset_radar_refl[:]-beginTimeOfCloud)).argmin()
endTimestepOfCloud = (abs(time_offset_radar_refl[:]-endTimeOfCloud)).argmin()
timestepRangeCloud = arange(beginTimestepOfCloud,endTimestepOfCloud)

if ( radarType == "arscl" ):
    height = radar_refl_nc.variables['height'].data
else:
    height = radar_refl_nc.variables['range'].data
    #range_gate_spacing = 29.979246
    #height = arange(0, 676*range_gate_spacing-1, range_gate_spacing)        

# Compute top and bottom grid levels for block of cloud for overlap analysis
cloudBaseLevel = (abs(height[:]-cloudBaseHeight)).argmin()
cloudTopLevel = (abs(height[:]-cloudTopHeight)).argmin()
levelRangeCloud = arange(cloudBaseLevel,cloudTopLevel)

lenTimestepRangeCloud = len(timestepRangeCloud)
lenLevelRangeCloud = len(levelRangeCloud)

if ( radarType == "arscl" ):
    # To extract the data part of the object, use [:,:] instead of data
    reflectivity_copol = radar_refl_nc.variables['reflectivity_best_estimate'].data
    # Pull the quality control flag from netcdf data
    qcRefl = radar_refl_nc.variables['qc_reflectivity_best_estimate'].data
    # Mask out reflectivity values outside of cloud or rain
    reflectivity_copol = masked_where( qcRefl > 0, reflectivity_copol )
else:
    reflectivity_copol = radar_refl_nc.variables['reflectivity_copol'].data

#pdb.set_trace()

# The numpy ix_ function is needed to extract the right part of the matrix
reflCopol = reflectivity_copol[ix_(time_range,height_range)]

# Block of cloud values for overlap analysis
reflCloudBlock = reflectivity_copol[ix_(timestepRangeCloud,levelRangeCloud)]

#pdb.set_trace()

# Replace small values with threshold in order to reduce the range of values to plot
#reflCopol = fmax(minThreshRefl,reflCopol[:])

#pdb.set_trace()

#reflCompressed = reflCopol[:,range_level].compressed()
reflCompressed = reflCloudBlock[:,range_level-cloudBaseLevel].compressed()

#dfser['ecdf_r']=(len(dfser)-dfser['rank']+1)/len(dfser)

lenReflCompressed = len(reflCompressed)

# Check whether there is cloud at the height level chosen for plotting time series
if ( len( reflCompressed ) == 0 ):
    print "ERROR: Reflectivity time series at level %s has no values above the threshold!!!" %range_level 
    sys.exit(1)

# Smallest and largest values of reflectivity, used for plots below
minRefl = min( reflCompressed )
maxRefl = max( reflCompressed )
reflRange = linspace(minRefl,maxRefl)

#pdb.set_trace()

# Compute effect of vertical overlap on radiation
# To do so, sum reflectivity from each profile, using original and sorted cloud data
# Compute the standard deviation in the sum
reflCloudBlockMin = amin(reflCloudBlock)
reflCloudBlockOffset = reflCloudBlock - reflCloudBlockMin
# Sum reflectivities in vertical, and then compute within-cloud mean
meanReflCloudBlock = mean(sum(reflCloudBlockOffset,axis=1))
reflCloudBlockFilled = filled(reflCloudBlockOffset,fill_value=0)
# Compute maximal overlap by sorting each altitude level individually
reflCloudBlockFilledSorted = zeros((lenTimestepRangeCloud,lenLevelRangeCloud))
for col in range(0,lenLevelRangeCloud):
    reflCloudBlockFilledSorted[:,col] = sort(reflCloudBlockFilled[:,col]) 
# Assertion check
if ( not( isclose( mean( mean(reflCloudBlockFilled) ), mean( mean(reflCloudBlockFilledSorted) ) ) ) ):
    print "ERROR: Computing maximal overlap failed!!! %s != %s" \
            % (mean( mean(reflCloudBlockFilled) ) , mean( mean(reflCloudBlockFilledSorted) ))  
    sys.exit(1)

#pdb.set_trace()
    
meanAlbedoUnsorted, LWPUnsorted \
                = calcMeanAlbedo(reflCloudBlockFilled, meanReflCloudBlock, meanLWP)

meanAlbedoSorted, LWPSorted \
                 = calcMeanAlbedo(reflCloudBlockFilledSorted, meanReflCloudBlock, meanLWP)

# Now consider a case in which there is no within-cloud variability
# mean within-cloud optical depth
tauWc0 = 0.15 * meanLWP
# mean within-cloud albedo
albedoWc0 = tauWc0 / (9.0 + tauWc0)
# Find cloud cover
sumReflCloudBlockFilled = sum(reflCloudBlockFilled,axis=1)
cloudCover = count_nonzero(sumReflCloudBlockFilled)/len(sumReflCloudBlockFilled)
# mean albedo, including clear air
meanAlbedo0 = albedoWc0 * cloudCover

# Assertion check
if ( not( isclose( meanReflCloudBlock * cloudCover , mean(sumReflCloudBlockFilled) ) ) ):
    print "ERROR: Computing maximal overlap failed!!! %s != %s" \
            % ( meanReflCloudBlock * cloudCover , mean(sumReflCloudBlockFilled) )  
    sys.exit(1)

#pdb.set_trace()

print "                                 Unsorted    Sorted   No within-cloud variability"
print "Mean Albedo:                     %.5f   %.5f   %.5f" %(meanAlbedoUnsorted,  meanAlbedoSorted, meanAlbedo0)
print "Relative fractional difference:  %.5f   %.5f   %.5f" \
       %( 0, 
        (meanAlbedoUnsorted-meanAlbedoSorted)/meanAlbedoUnsorted,
        (meanAlbedo0-meanAlbedoSorted)/meanAlbedoUnsorted
         )

# Compute Spearman's rank correlation matrix 
# among reflectivity at different vertical levels.
# I'm not sure about the following calculation because 
# I don't understand a correlation of a masked array.
spearmanMatrix, spearmanPval = spearmanr(reflCloudBlock, axis=0)

print "Spearman rank correlation matrix:"
print spearmanMatrix

#exit
TIME, HEIGHT = meshgrid(height[height_range], 
                        time_offset_radar_refl[time_range]) 
plt.ion() # Use interactive mode so that program continues when plot appears
plt.clf()
# either contourf or pcolormesh produces filled contours
radarContour = plt.pcolormesh(HEIGHT[:],TIME[:],reflCopol)
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = plt.colorbar(radarContour)
cbar.ax.set_ylabel('Reflectivity  [dBZ]')
# Add the contour line levels to the colorbar
#cbar.add_lines(radarContour)
# Plot horizontal line corresponding to time series plot later
plt.plot( [ beginTimeOfPlot , endTimeOfPlot  ],  
          [ height[range_level], height[range_level] ], 'k' )
#pdb.set_trace()
# Plot vertical line corresponding to histogram of reflectivity
plt.plot( [ time_of_cloud , time_of_cloud  ],  
          [ height[height_range[0]], height[height_range[len(height_range)-1]]  ], 'k' )
# Plot box corresponding to cloud box
plt.plot( [ beginTimeOfCloud , beginTimeOfCloud ],  [ cloudBaseHeight, cloudTopHeight ], 'k' )
plt.plot( [ endTimeOfCloud , endTimeOfCloud ],  [ cloudBaseHeight, cloudTopHeight ], 'k' )
plt.plot( [ beginTimeOfCloud , endTimeOfCloud ],  [ cloudBaseHeight, cloudBaseHeight ], 'k' )
plt.plot( [ beginTimeOfCloud , endTimeOfCloud ],  [ cloudTopHeight, cloudTopHeight ], 'k' )

plt.title('Radar reflectivity')
plt.xlabel('Time  [' + time_offset_radar_refl.units + ']')
plt.ylabel('Altitude  [m]')
plt.figure()
#plt.show()

#pdb.set_trace()


# uniformCloudBlock = close-up selection of contiguous cloud values.  
# Each column is a different altitude.
uniformCloudBlock = zeros((lenTimestepRangeCloud,lenLevelRangeCloud))
for col in range(0,lenLevelRangeCloud):
    uniformCloudBlock[:,col] = rankdata(reflCloudBlock[:,col]) /    \
                                MaskedArray.count(reflCloudBlock[:,col])

uniformCloudBlock = masked_where( uniformCloudBlock == 0, uniformCloudBlock )
# I'm not sure if it's appropriate to rank first, then fill.
# So I'm not sure if this is correct.
uniformCloudBlockFilled = filled(uniformCloudBlock,fill_value=0)

plt.clf()
for idx in range(1,5):
    plt.subplot(2,2,idx)
    plt.plot(uniformCloudBlockFilled[:,5],uniformCloudBlockFilled[:,idx],'.')
plt.title('Copula')
plt.figure()

#pdb.set_trace()
#plt.ion() # Use interactive mode so that program continues when plot appears
plt.clf()
plt.subplot(121)
#pdb.set_trace()
for idx in range(0,numProfiles):
    plt.plot(reflCopol[timestep_of_cloud-beginTimestepOfPlot+idx,levelRangeCloud],
             height[levelRangeCloud],'-o')
plt.ylim(height[levelRangeCloud[0]], height[levelRangeCloud[len(levelRangeCloud)-1]])
plt.xlabel('Copolar radar reflectivity')
plt.ylabel('Altitude  [m]')

plt.subplot(122)
for idx in range(0,numProfiles):
    plt.plot(uniformCloudBlock[timestep_of_cloud-beginTimestepOfCloud+idx,:],
             height[levelRangeCloud],'-o')
plt.ylim(height[levelRangeCloud[0]], height[levelRangeCloud[len(levelRangeCloud)-1]])
plt.xlabel('Uniform distribution of reflectivity')
plt.ylabel('Altitude  [m]')
plt.figure()

plt.subplot(121)
for idx in range(0,numProfiles):
    plt.plot(spearmanMatrix[:,idx],
             height[levelRangeCloud],'-o')
plt.ylim(height[levelRangeCloud[0]], height[levelRangeCloud[len(levelRangeCloud)-1]])
plt.xlabel('Spearman rank correlations  [-]')
plt.ylabel('Altitude  [m]')
plt.figure()


#plt.clf()
##pdb.set_trace()
#for idx in range(0,lenTimestepRangeCloud):
#    plt.plot(uniformCloudBlock[idx,:],
#             height[levelRangeCloud],'-o')
#plt.xlabel('Copolar radar reflectivity')
#plt.ylabel('Altitude  [m]')
#plt.figure()




plt.clf()
TIMECLD, HEIGHTCLD = meshgrid(height[levelRangeCloud], 
                        time_offset_radar_refl[timestepRangeCloud]) 
# either contourf or pcolormesh produces filled contours
uniformCloudBlockContour = plt.pcolormesh(HEIGHTCLD[:],TIMECLD[:],uniformCloudBlock)
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = plt.colorbar(uniformCloudBlockContour)
cbar.ax.set_ylabel('Normalized Reflectivity  []')
# Add the contour line levels to the colorbar
#cbar.add_lines(radarContour)
# Plot horizontal line corresponding to time series plot later
plt.plot( [ beginTimeOfCloud , endTimeOfCloud  ],  
          [ height[range_level], height[range_level] ], 'k' )
#pdb.set_trace()
# Plot vertical line corresponding to histogram of reflectivity
plt.plot( [ time_of_cloud , time_of_cloud  ],  
          [ height[levelRangeCloud[0]], height[levelRangeCloud[len(levelRangeCloud)-1]]  ], 'k' )
plt.title('Normalized reflectivity')
plt.xlabel('Time  [' + time_offset_radar_refl.units + ']')
plt.ylabel('Altitude  [m]')
plt.figure()
                        
radar_refl_nc.close()

# Compute mean and variance of truncated time series
truncMean = mean( reflCompressed )
truncVarnce = var( reflCompressed )
print "truncMean = %s"  %truncMean
print "sqrt(truncVarnce) = %s"  %sqrt(truncVarnce)

# Compute parameters of truncated normal distribution
muInit = 2*truncMean 
sigmaInit = 2*sqrt(truncVarnce)
mu, sigma = findTruncNormalRoots(truncMean,truncVarnce,muInit,sigmaInit,minThreshRefl)
print "mu = %s" %mu
print "sigma = %s" %sigma

# Compute empirical distribution function of data
reflCompressedSorted = sort(reflCompressed)
reflEdf = (rankdata(reflCompressedSorted) - 1)/lenReflCompressed
normCdf = norm.cdf( reflCompressedSorted, loc=truncMean, scale=sqrt(truncVarnce) )
truncNormCdf = ( norm.cdf(reflCompressedSorted,mu,sigma) \
                                      - norm.cdf((minRefl-mu)/sigma) ) \
                          /(1.0-norm.cdf((minRefl-mu)/sigma))
minRefl = amin(reflCompressedSorted)
expMuLogN = (truncMean-minRefl)/sqrt(1+truncVarnce/((truncMean-minRefl)**2))
sigma2LogN = log(1+truncVarnce/((truncMean-minRefl)**2))
lognormCdf = lognorm.cdf( reflCompressedSorted - minRefl, sqrt(sigma2LogN),
                      loc=0, scale=expMuLogN )

#pdb.set_trace()

DnNormCdf = findKSDn(normCdf, reflEdf)
DnTruncNormCdf = findKSDn(truncNormCdf, reflEdf)
DnLognormCdf = findKSDn(lognormCdf, reflEdf)
print "KS statistic Dn"
print "DnNormCdf = %s" %DnNormCdf
print "DnTruncNormCdf = %s" %DnTruncNormCdf
print "DnLognormCdf = %s" %DnLognormCdf

plt.clf()

# Plot cumulative distribution functions
# Empirical CDF
plt.plot( reflCompressedSorted , reflEdf, label="Empirical" )
# Normal CDF
plt.plot( reflCompressedSorted , 
          normCdf ,
          label="Normal"   )
# Truncated normal CDF
truncNormCurve = plt.plot( reflCompressedSorted, 
                          truncNormCdf ,
                          label="Truncated normal")
# Lognormal CDF
plt.plot( reflCompressedSorted , 
          lognormCdf ,
          label="Lognorm" )
plt.xlabel('Copolar radar reflectivity')
plt.ylabel('Cumulative distribution function')
plt.title('Height = %s m' %height[range_level] )
plt.legend(loc="best")
plt.figure()

# Plot time series of radar reflectivity
plt.clf()
plt.subplot(211)
plt.plot(time_offset_radar_refl[time_range],reflCopol[:,range_level].data)
plt.ylim((minThreshRefl,1.1*max(reflCopol[:,range_level]).data))
plt.xlabel('Time [' + time_offset_radar_refl.units + ']')
plt.ylabel('Copolar radar reflectivity')
plt.title('Height = %s m.  Stdev. = %s dBZ. Sk = %s' \
            % ( height[range_level], std(reflCompressed), skew(reflCompressed) )  )

#pdb.set_trace()

# Plot histogram of copolar radar reflectivity
plt.subplot(212)
n, bins, patches = plt.hist(reflCompressed, 
                            50, normed=True, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
# Overplot best-fit truncated normal
truncNormCurve = plt.plot(reflRange, 
                          norm.pdf(reflRange,loc=mu,scale=sigma)/(1.0-norm.cdf((minRefl-mu)/sigma)), 
                          label="Truncated normal")
# Overplot best-fit normal
normCurve = plt.plot( reflRange, 
                     norm.pdf(reflRange,loc=truncMean,scale=sqrt(truncVarnce)) , 
                     label="Normal" )
# Overplot best-fit lognormal
plt.plot( reflRange , 
          lognorm.pdf( reflRange - minRefl, sqrt(sigma2LogN),
                      loc=0, scale=expMuLogN )  , label="Lognormal" )
plt.xlabel('Copolar radar reflectivity')
plt.ylabel('Probability')
plt.legend()

#plt.show()
plt.draw()

#plt.close()

#exit