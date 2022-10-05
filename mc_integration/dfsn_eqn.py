# -*- coding: utf-8 -*-
"""  A function that integrates a nonlinear 1D diffusion """

import pdb

# pdb.set_trace()


def calcMean(K, qt_init, dz, nz, dt, nt):
    """
    Solve a diffusion equation for the grid-mean total water, qt_alltimes,
    given an initial profile, qt_init.
    Assume zero moisture flux at the top and bottom of the domain.
    
    Reference: http://hplgit.github.io/num-methods-for-PDEs/doc/pub/diffu/sphinx/._main_diffu001.html#forward-euler-scheme
    Reference: http://hplgit.github.io/num-methods-for-PDEs/doc/pub/nonlin/sphinx/._main_nonlin004.html#finite-difference-discretization-1

    nz = number of full vertical grid levels
    K = Eddy diffusivity,  interface levels, 0 to nz
    qt_init = initial qt profile, full levels, 0 to nz-1
    dt = time step intervals
    nt = number of time steps
    """

    import numpy as np
    import pdb

    qt_n = qt_init  # Indices go from 0 to nz-1
    qt_np1 = np.zeros_like(qt_init)

    # Save output at all time steps
    qt_alltimes = np.zeros((nz,nt))
    qt_alltimes[:,0] = qt_init

    print("qt_alltimes[:,0] = ", qt_alltimes[:,0])

    for n in range(0, nt-1):
        # Compute qt in interior
        for i in range(1, nz-1):
            qt_np1[i] = qt_n[i] + (dt/(dz*dz)) * \
                ( K[i+1]*(qt_n[i+1]-qt_n[i]) - K[i]*(qt_n[i]-qt_n[i-1]) )

        # Apply no-flux boundaries at top and bottom
        qt_np1[0] = qt_np1[1]
        qt_np1[nz-1] = qt_np1[nz-2]

        qt_alltimes[:,n+1] = qt_np1

        # Update field
        qt_n = qt_np1

        #pdb.set_trace()
    
    return qt_alltimes
 
def calcVariance(K, qt_alltimes, dz, nz, nt):
    """
    Diagnose the variance of total water, qt2_alltimes, given
    the grid-mean profile of total water, qt_alltimes. 
    
    Reference: http://hplgit.github.io/num-methods-for-PDEs/doc/pub/diffu/sphinx/._main_diffu001.html#forward-euler-scheme
    Reference: http://hplgit.github.io/num-methods-for-PDEs/doc/pub/nonlin/sphinx/._main_nonlin004.html#finite-difference-discretization-1
    """
    import numpy as np

    # Eddy turnover time
    tau = 0.01

    # Initialize variance of total water
    qt2_alltimes = np.zeros((nz+1,nt))

    for n in range(0, nt):
        # Compute qt2 away from boundaries
        for i in range(1, nz):
            qt2_alltimes[i,n] = tau * K[i] * \
                ( (qt_alltimes[i,n]-qt_alltimes[i-1,n]) / dz )**2
    
    return qt2_alltimes
   
def main():
    """ Solve a minimal closure model.   Namely, integrate forward in time 
    a prognostic equation for total water and a diagnostic equation 
    for total water variance.  
    """
    import numpy as np 
    import math
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    #  The model uses a staggered grid.
    #  Vertical grid levels
    #  Interface levs z=1 -------------------  qt2(nz), K(nz)
    #  Full levels        - - - - - - - - - -  qt(nz-1)
    #                     -------------------  qt2, K
    #                       .    .    .
    #                     - - - - - - - - - -  qt(1)    
    #                     -------------------  qt2(1), K(1)
    #                     - - - - - - - - - -  qt(0)
    #                 z=0 -------------------  qt2(0), K(0)
    #                     ///////////////////


    nz = 100  # Number of full (interior) vertical grid levels (0 to nz-1)
    dz = 1/nz   # Vertical grid spacing

    dt = 0.00001    # Time step duration
    nt = 1000     # Number of time steps
    
    # Set up altitude of vertical grid levels.
    # The domain runs from z=0 to z=1.
    zIntLevs = np.linspace(0, 1, nz+1)
    zFullLevs = zIntLevs - 0.5*dz
    zFullLevs = zFullLevs[1:] # Now zFullLevs goes from 0 to nz-1

    # Define eddy diffusivity, K
    gridUpperHalfInt = list(range(math.ceil(nz/2),nz+1))
    K = np.ones_like(zIntLevs)
    K[gridUpperHalfInt] = zIntLevs[gridUpperHalfInt] + 0.5

    # Define initial condition for mean total water, qt
    gridUpperHalfFull = list(range(math.ceil(nz/2),nz))
    qt_sfc = 10.
    qt_init = qt_sfc * np.ones_like(zFullLevs)
    qt_init[gridUpperHalfFull] = \
        -qt_sfc * ( zIntLevs[gridUpperHalfFull] - 0.5 ) + qt_sfc
    qt_init[nz-1] = qt_init[nz-2] #  No-flux upper boundary condition

    print("zIntLevs =", zIntLevs)
    print("K = ", K)
    print("qt_init =", qt_init)
    print("qt_init[0] =", qt_init[0])


    # Calculate the profile of mean total water, qt, 
    #    at all times.
    qt_alltimes = calcMean(K, qt_init, dz, nz, dt, nt)

    # Calculate the profile of total water variance, qt2,
    #    at all times.
    qt2_alltimes = calcVariance(K, qt_alltimes, dz, nz, nt)
    
    
    plt.clf()

    plt.subplot(221)
    plt.plot(qt_alltimes[:,0], zFullLevs, label='qt(t=0)')    
    plt.legend()
    plt.xlabel('qt(t=0)')
    plt.ylabel('Altitude, z')

#    plt.subplot(221)
#    plt.plot(K, zIntLevs, label='Eddy diffusivity, K')    
#    plt.legend()
#    plt.xlabel('Eddy diffusivity, K')
#    plt.ylabel('Altitude, z')

    plt.subplot(222)
    plt.plot(qt_alltimes[:,nt-1], zFullLevs, label='qt(nt-1)')    
    plt.legend()
    #plt.xlabel('qt')
    plt.ylabel('Altitude, z')

    plt.subplot(223)
    plt.plot(qt2_alltimes[:,nt-1], zIntLevs, label='qt2(nt-1)')
    plt.legend()    
    plt.xlabel('qt2')
    plt.ylabel('Altitude, z')

    # plt.subplot(224)
    # plt.contourf(qt2_alltimes)
    # plt.colorbar()    
    # plt.xlabel('time')
    # plt.ylabel('Altitude, z')

    plt.subplot(224)
    #sigma = math.sqrt(qt2_alltimes[math.ceil(0.7*nz),math.ceil(0.7*nt)])
    sigma = np.sqrt(qt2_alltimes[math.ceil(0.7*nz),:])
    #mu = qt_alltimes[math.ceil(0.7*nz),math.ceil(0.7*nt)]
    mu = qt_alltimes[math.ceil(0.7*nz),:]
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))


    # plt.subplot(224)
    # plt.plot(qt_alltimes[:,1], zFullLevs, label='qt(n=1)')    
    # plt.legend()
    # plt.xlabel('qt(t=1)')
    # plt.ylabel('Altitude, z')

    # plt.subplot(224)
    # plt.plot(qt_alltimes[:,1], zFullLevs, label='qt(n=1)')    
    # plt.legend()
    # plt.xlabel('qt(t=1)')
    # plt.ylabel('Altitude, z')

    # plt.subplot(224)
    # plt.plot(qt2_alltimes[:,nt-1], zIntLevs, label='qt2')    
    # plt.legend()
    # plt.xlabel('qt2')
    # plt.ylabel('Altitude, z')

    plt.show()    
    
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()

    