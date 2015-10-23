# -*- coding: utf-8 -*-
"""
   Analyze ARM observations.  
"""

# Import libraries
import numpy as np
import pylab
import matplotlib.pyplot as plt
from scipy.io import netcdf

# Point to directory containing ARM observations 
data_dir = '/home/studi/Larson/arm_obs/arm_data_files'

data_file = '/home/studi/Larson/arm_data_files/sgpbeflux1longC1.c1.20131215.000000.custom.cdf'
#data_file = '/home/studi/Larson/arm_data_files/sgpbeflux1longC1.c1.20131217.000000.custom.cdf'
#data_file = '/home/studi/Larson/arm_data_files/sgpbeflux1longC1.c1.20131218.000000.custom.cdf'

#data_file = 'simple_xy.nc'

nc_file = netcdf.netcdf_file(data_file, 'r')


time_offset = nc_file.variables['time_offset']
short_direct_normal = nc_file.variables['short_direct_normal']

sdn_truncated = [x for x in short_direct_normal if x > 10]

n, bins, patches = pylab.hist(sdn_truncated[:], 50, normed=1, histtype='stepfilled')
pylab.setp(patches, 'facecolor', 'g', 'alpha', 0.75)

# add a line showing the expected distribution
#y = pylab.normpdf( bins, mu, sigma)
#l = pylab.plot(bins, y, 'k--', linewidth=1.5)


#
# create a histogram by providing the bin edges (unequally spaced)
#
pylab.figure()
pylab.show()

#print(data[1:10])