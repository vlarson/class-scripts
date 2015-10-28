# -*- coding: utf-8 -*-
"""
   Analyze ARM observations.  
"""

# Import libraries
from numpy import fmax, arange, meshgrid, array, ix_
import matplotlib.pyplot as plt
from scipy.io import netcdf
import pdb


# Point to directory containing ARM observations 
data_dir = '/home/studi/Larson/arm_data_files/'

##################################
#
#  Plot surface radiative fields
#
##################################

#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131204.000000.custom.cdf'
beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131205.000000.custom.cdf'
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131206.000000.custom.cdf'
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131207.000000.custom.cdf'
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131208.000000.custom.cdf'
# SDN showed a few clouds on 20131215:
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131215.000000.custom.cdf'
# 20131217 had essentially clear skies
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131217.000000.custom.cdf'
#beflux_file = data_dir + 'sgpbeflux1longC1.c1.20131218.000000.custom.cdf'

beflux_nc = netcdf.netcdf_file(beflux_file, 'r')

time_offset_beflux = beflux_nc.variables['time_offset']
short_direct_normal = beflux_nc.variables['short_direct_normal']

# Impose a threshold on short_direct_normal to get rid of nighttime values
min_sdn = 10

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
#pylab.figure()
#pylab.show()

#pdb.set_trace()

# Plot histogram of shortwave direct normal flux
#plt.clf()
plt.subplot(212)
n, bins, patches = plt.hist(sdn_truncated[:], 50, normed=1, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
plt.xlabel('Shortwave direct normal flux')
plt.ylabel('Probability')
plt.figure()
plt.show()

##################################
#
#  Plot radar obs
#
##################################

#date = 20131205
#date = 20131206
#date = 20131207
date = 20131215

if date == 20131205:
    # Radar could see clouds up to 8 km on 20131205:
    radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131205.000002.custom.nc'
    # Grid level at which to plot time series and histogram    
    range_level = 167  
    # Time and time step at which profile of reflectivity is plotted
    time_of_cloud = 69000 
    # Impose a threshold on reflectivity_copol to get rid of nighttime values
    min_refl = -30
    # Indices for range of altitudes for time-height plots
    height_range = arange(50,250)
    # Indices for range of times for time-height plots
    time_range_half_width = 2000
elif date == 20131206:
    # Radar could see clouds up to 2 km on 20131206:
    radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131206.000000.custom.nc'    
    # Grid level at which to plot time series and histogram
    range_level = 45
    # Time and time step at which profile of reflectivity is plotted
    time_of_cloud = 75000 
    # Impose a threshold on reflectivity_copol to get rid of nighttime values
    min_refl = -30
    # Indices for range of altitudes for time-height plots
    height_range = arange(0,100)
    # Indices for range of times for time-height plots
    time_range_half_width = 1000
elif date == 20131207:
    # Radar could see clouds up to 4 km on 20131207:
    radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131207.000001.custom.nc'    
    # Grid level at which to plot time series and histogram
    range_level = 117
    # Time and time step at which profile of reflectivity is plotted
    time_of_cloud = 69000 
    # Impose a threshold on reflectivity_copol to get rid of nighttime values
    min_refl = -30
    # Indices for range of altitudes for time-height plots
    height_range = arange(0,200)
    # Indices for range of times for time-height plots
    time_range_half_width = 2000
elif date == 20131215:
    # Radar could see clouds up to 4 km on 20131207:
    radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131215.000003.custom.nc'    
    # Grid level at which to plot time series and histogram
    range_level = 117
    # Time and time step at which profile of reflectivity is plotted
    time_of_cloud = 69000 
    # Impose a threshold on reflectivity_copol to get rid of nighttime values
    min_refl = -30
    # Indices for range of altitudes for time-height plots
    height_range = arange(0,200)
    # Indices for range of times for time-height plots
    time_range_half_width = 2000
else:
    print "Wrong date"

#radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131204.000000.custom.nc'
# Radar could see clouds up to 8 km on 20131205:
#radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131205.000002.custom.nc'
# Radar could see clouds up to 2 km on 20131206:
#radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131206.000000.custom.nc'
# Radar could see clouds up to 4 km on 20131207:
#radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131207.000001.custom.nc'
#radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131208.000003.custom.nc'
# Radar couldn't see clouds on 20131215:
#radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131215.000003.custom.nc'
# 20131217 had essentially clear skies
#radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131217.000003.custom.nc'
#radar_refl_file = data_dir + 'sgpkazrcorgeC1.c1.20131218.000001.custom.nc'

radar_refl_nc = netcdf.netcdf_file(radar_refl_file, 'r')

time_offset_radar_refl = radar_refl_nc.variables['time_offset']

# The final [:,:] converts to a numpy array, I think
#reflectivity_copol = radar_refl_nc.variables['reflectivity_copol'][:,:]
reflectivity_copol = radar_refl_nc.variables['reflectivity_copol'].data


# Impose a threshold on reflectivity_copol to get rid of nighttime values
#min_refl = -30

# Grid level at which to plot time series and histogram
#range_level = 167  #  Level of clouds in 20131205
#range_level = 45    #  Level of clouds in 20131206
#range_level = 117  #  Level of clouds in 20131207
#range_level = 45    

# Time and time step at which profile of reflectivity is plotted
#time_of_cloud = 69000 # Time of clouds in 20131205
#time_of_cloud = 75000 # Time of clouds in 20131206
#time_of_cloud = 69000 # Time of clouds in 20131207
#time_of_cloud = 69000

timestep_of_cloud = (abs(time_offset_radar_refl[:]-time_of_cloud)).argmin()

#pdb.set_trace()

# Replace small values with threshold, for plotting time series
refl_floored = fmax(min_refl,reflectivity_copol[:])

# Remove small values from time series, thereby shortening vector length
refl_truncated = [x for x in reflectivity_copol[:,range_level] if x > min_refl]

#pdb.set_trace()

# Plot time series of shortwave direct normal flux
plt.clf()
plt.subplot(211)
plt.plot(time_offset_radar_refl[:],refl_floored[:,range_level])
plt.xlabel('Time')
plt.ylabel('Copolar radar reflectivity')
#pylab.figure()
#pylab.show()

#pdb.set_trace()

# Plot histogram of copolar radar reflectivity
#plt.clf()
plt.subplot(212)
n, bins, patches = plt.hist(refl_truncated[:], 50, normed=1, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
plt.xlabel('Copolar radar reflectivity')
plt.ylabel('Probability')
plt.figure()
plt.show()

plt.clf()

range_gate_spacing = 29.979246
#time = arange(-3.0, 3.0, delta)
height = arange(0, 676*range_gate_spacing-1, range_gate_spacing)
plt.plot(reflectivity_copol[timestep_of_cloud,:],height[:])
plt.xlabel('Copolar radar reflectivity')
plt.ylabel('Altitude  [m]')
plt.figure()
plt.show

#pdb.set_trace()

#height_range = arange(50,250)
#time_range_half_width = 2000
time_range = arange((timestep_of_cloud-time_range_half_width),
                    (timestep_of_cloud+time_range_half_width))

#exit
TIME, HEIGHT = meshgrid(height[height_range], 
                        time_offset_radar_refl[time_range])
# The numpy ix_ function is needed to extract the right part of the matrix 
# either contourf or pcolormesh produces filled contours
radarContour = plt.pcolormesh(HEIGHT[:],TIME[:],reflectivity_copol[ix_(time_range,height_range)])
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
#
#plt.clf()
#range_gate_spacing = 29.979246
##time = arange(-3.0, 3.0, delta)
#height = arange(0, 676*range_gate_spacing-1, range_gate_spacing)
#plt.plot(height[:],reflectivity_copol[16000,:])
##plt.figure()
##plt.show()
#
##pdb.set_trace()
                        
radar_refl_nc.close()