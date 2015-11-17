# -*- coding: utf-8 -*-
"""
   Analyze ARM observations.  
"""

# Import libraries
from numpy import fmax, arange, meshgrid, ix_, sqrt, mean, var, linspace, asarray
from numpy.ma import masked_where
from math import pi
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.io import netcdf
from arm_utilities import plotSfcRad, findTruncNormalRoots
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
# Time and time step at which profile of reflectivity is plotted
time_of_cloud = 69000 
# Indices for range of altitudes for time-height plots
height_range = arange(0,200)
# Indices for range of times for time-height plots
beginTimeOfPlot = 1
endTimeOfPlot = 86399
# If lArscl = True, then we want to use the ARSCL radar retrieval
radarType = "arscl"
# Impose a threshold on reflectivity_copol to get rid of noise values
if (radarType == "arscl"):
    minThreshRefl = -60
else:
    minThreshRefl = -30
    
    
# Now overwrite defaults with specialized values for particular days

#date = 20131204    # Shallow Sc
#date = 20131205    # Deep Cu
#date = 20131206    # Shallow Sc, bad data?
#date = 20131207    # Sc/Cu from 3 to 4 km
#date = 20131208    # Low drizzling Cu
#date = 20131215    # No clouds
#date = 20131217    # Noise
#date = 20150607     # Shallow Cu and some mid level clouds
#date = 20150609     # Shallow Cu
date = 20150627     # Shallow Cu

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
    range_level = 80 
    # Indices for range of altitudes for time-height plots
    height_range = arange(0,250)
    # Time and time step at which profile of reflectivity is plotted
    time_of_cloud = 43200
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
    time_of_cloud = 43200
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
    time_of_cloud = 43200
else:
    print "Wrong date"


radar_refl_nc = netcdf.netcdf_file(radar_refl_file, 'r')

time_offset_radar_refl = radar_refl_nc.variables['time_offset']

#timestep_of_cloud = (abs(time_offset_radar_refl[:]-time_of_cloud)).argmin()
beginTimestepOfPlot = (abs(time_offset_radar_refl[:]-beginTimeOfPlot)).argmin()
endTimestepOfPlot = (abs(time_offset_radar_refl[:]-endTimeOfPlot)).argmin()

time_range = arange(beginTimestepOfPlot,endTimestepOfPlot)

if ( radarType == "arscl" ):
    height = radar_refl_nc.variables['height'].data
else:
    height = radar_refl_nc.variables['range'].data
    #range_gate_spacing = 29.979246
    #height = arange(0, 676*range_gate_spacing-1, range_gate_spacing)        

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
reflectivity_copol = reflectivity_copol[ix_(time_range,height_range)]

#pdb.set_trace()

# Replace small values with threshold in order to reduce the range of values to plot
#reflectivity_copol = fmax(minThreshRefl,reflectivity_copol[:])

#pdb.set_trace()

# Check whether there is cloud at the height level chosen for plotting time series
if ( len( reflectivity_copol[:,range_level].compressed() ) == 0 ):
    print "ERROR: Reflectivity time series at level %s has no values above the threshold!!!" %range_level 
    sys.exit(1)

#pdb.set_trace()

#plt.clf()
#
#
#plt.plot(reflectivity_copol[timestep_of_cloud,:],height[:])
#plt.xlabel('Copolar radar reflectivity')
#plt.ylabel('Altitude  [m]')
#plt.figure()
#plt.show

#pdb.set_trace()

#exit
TIME, HEIGHT = meshgrid(height[height_range], 
                        time_offset_radar_refl[time_range])
# The numpy ix_ function is needed to extract the right part of the matrix 
# either contourf or pcolormesh produces filled contours
plt.clf()
radarContour = plt.pcolormesh(HEIGHT[:],TIME[:],reflectivity_copol)
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = plt.colorbar(radarContour)
# Plot horizontal line corresponding to time series plot later
plt.plot( [ beginTimeOfPlot , endTimeOfPlot  ],  
          [ height[range_level], height[range_level] ], 'k' )
cbar.ax.set_ylabel('Reflectivity  [dBZ]')
# Add the contour line levels to the colorbar
#cbar.add_lines(radarContour)
plt.title('Radar reflectivity')
plt.xlabel('Time  [' + time_offset_radar_refl.units + ']')
plt.ylabel('Altitude  [m]')
plt.figure()
#plt.show()

#pdb.set_trace()
                        
radar_refl_nc.close()

# Compute mean and variance of truncated time series
truncMean = mean( reflectivity_copol[:,range_level].compressed() )
truncVarnce = var( reflectivity_copol[:,range_level].compressed() )

muInit = truncMean 
sigmaInit = sqrt(truncVarnce)

print "truncMean = %s"  %truncMean
print "sqrt(truncVarnce) = %s"  %sqrt(truncVarnce)

mu, sigma = findTruncNormalRoots(truncMean,truncVarnce,muInit,sigmaInit,minThreshRefl)

print "mu = %s" %mu
print "sigma = %s" %sigma


# Plot time series of radar reflectivity
plt.clf()
plt.subplot(211)
plt.plot(time_offset_radar_refl[time_range],reflectivity_copol[:,range_level].data)
plt.ylim((minThreshRefl,1.1*max(reflectivity_copol[:,range_level]).data))
plt.xlabel('Time [' + time_offset_radar_refl.units + ']')
plt.ylabel('Copolar radar reflectivity')
plt.title('Height = %s m' %height[range_level] )

#pdb.set_trace()

# Plot histogram of copolar radar reflectivity
plt.subplot(212)
n, bins, patches = plt.hist(reflectivity_copol[:,range_level].compressed(), 
                            50, normed=True, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
plt.xlabel('Copolar radar reflectivity')
plt.ylabel('Probability')
minRefl = min( reflectivity_copol[:,range_level].compressed() )
maxRefl = max( reflectivity_copol[:,range_level].compressed() )
reflRange = linspace(minRefl,maxRefl)
normCurve = plt.plot(reflRange, norm.pdf(reflRange,mu,sigma)/(1.0-norm.cdf((minRefl-mu)/sigma)))
plt.figure()
plt.show()

#plt.close()

#exit