# -*- coding: utf-8 -*-
"""
   Analyze ARM observations.  
"""

# Import libraries
from numpy import fmax, arange, meshgrid, ix_, sqrt, mean, var, linspace, asarray
from math import pi
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.io import netcdf
from arm_utilities import plotSfcRad, findTruncNormalRoots
import pdb


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
# Impose a threshold on reflectivity_copol to get rid of nighttime values
minThreshRefl = -30
# Indices for range of altitudes for time-height plots
height_range = arange(0,200)
# Indices for range of times for time-height plots
time_range_half_width = 2000

# Now overwrite defaults with specialized values for particular days

#date = 20131204    # Shallow Sc
#date = 20131205    # Deep Cu
#date = 20131206    # Shallow Sc, bad data?
#date = 20131207    # Sc/Cu from 3 to 4 km
date = 20131208    # Low drizzling Cu
#date = 20131215    # No clouds
#date = 20131217    # Noise

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
    # Indices for range of times for time-height plots
    time_range_half_width = 4000
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
else:
    print "Wrong date"


radar_refl_nc = netcdf.netcdf_file(radar_refl_file, 'r')

time_offset_radar_refl = radar_refl_nc.variables['time_offset']

timestep_of_cloud = (abs(time_offset_radar_refl[:]-time_of_cloud)).argmin()

time_range = arange((timestep_of_cloud-time_range_half_width),
                    (timestep_of_cloud+time_range_half_width))
                    
# The final [:,:] converts to a numpy array, I think.
#reflectivity_copol = radar_refl_nc.variables['reflectivity_copol'][:,:]
# Or just use .data, as follows:
reflectivity_copol = radar_refl_nc.variables['reflectivity_copol'].data

# The numpy ix_ function is needed to extract the right part of the matrix
reflectivity_copol = reflectivity_copol[ix_(time_range,height_range)]

#pdb.set_trace()

# Replace small values with threshold, for plotting time series
refl_floored = fmax(minThreshRefl,reflectivity_copol[:])

# Remove small values from time series, thereby shortening vector length,
# and convert to numpy array
reflTrunc = asarray([x for x in reflectivity_copol[:,range_level] if x > minThreshRefl])

range_gate_spacing = 29.979246
height = arange(0, 676*range_gate_spacing-1, range_gate_spacing)

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
cbar.ax.set_ylabel('Reflectivity  [dBZ]')
# Add the contour line levels to the colorbar
#cbar.add_lines(radarContour)
plt.title('Radar reflectivity')
plt.xlabel('Time')
plt.ylabel('Altitude  [m]')
plt.figure()
##plt.show()
                        
radar_refl_nc.close()

# Compute mean and variance of truncated time series
truncMean = mean(reflTrunc)
truncVarnce = var(reflTrunc)

## Unit test: should return mu=0, sigma=1
#truncMean = 2.0 / sqrt(2*pi)
#truncVarnce = 1.0 - 4.0/(2.0*pi)
#minThreshRefl = 0

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
plt.plot(time_range,refl_floored[:,range_level])
plt.xlabel('Time')
plt.ylabel('Copolar radar reflectivity')

#pdb.set_trace()

# Plot histogram of copolar radar reflectivity
plt.subplot(212)
n, bins, patches = plt.hist(reflTrunc[:], 50, normed=True, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
plt.xlabel('Copolar radar reflectivity')
plt.ylabel('Probability')
minRefl = min(reflTrunc[:])
maxRefl = max(reflTrunc[:])
reflRange = linspace(minRefl,maxRefl)
normCurve = plt.plot(reflRange, norm.pdf(reflRange,mu,sigma)/(1.0-norm.cdf((minRefl-mu)/sigma)))
plt.figure()
plt.show()
