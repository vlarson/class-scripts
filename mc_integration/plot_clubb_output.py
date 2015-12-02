# -*- coding: utf-8 -*-
"""
   Plot netcdf output from CLUBB.  
"""

# Import libraries
import matplotlib.pyplot as plt
from scipy.io import netcdf
from numpy import squeeze
import pdb

# Point to directory containing CLUBB output
data_dir = '/home/studi/Larson/clubb/output/'
clubb_nc_file = data_dir + 'rico_lh_zt.nc'

timestep = 800



clubb_nc = netcdf.netcdf_file(clubb_nc_file, 'r')

time = clubb_nc.variables['time']

altitude = clubb_nc.variables['altitude'].data

rrainm = squeeze( clubb_nc.variables['rrm'].data )

#pdb.set_trace()

plt.ion() # Use interactive mode so that program continues when plot appears
plt.clf()
plt.subplot(121)
plt.plot( rrainm[timestep,:], altitude )
plt.xlabel('Rain mixing ratio   [kg/kg]')
plt.ylabel('Altitude  [m]')
plt.ylim(0,altitude[len(altitude)-1])
plt.draw()
