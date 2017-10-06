import numpy as np
import h5py
from scipy.integrate import nquad

def rho_init(x):
    return rho_initial

rho_init = np.vectorize(rho_init)

def theta_init(x):
    return temp_initial

theta_init = np.vectorize(theta_init)

def theta_walls(x):
    return temp_walls

theta_walls = np.vectorize(theta_walls)

def f0(v, x):
    # Setting the temperature corresponding to that of the wall beyond boundaries
    if(x<=0):
        theta = theta_walls(x)
        return 1 * (m/(2*np.pi*k*theta))**(1./2.) * np.exp(-m*v**2./(2.*k*theta))
    
    if(x>=1):
        theta = theta_walls(x)
        return 1 * (m/(2*np.pi*k*theta))**(1./2.) * np.exp(-m*v**2./(2.*k*theta))
    
    else:   
        rho   = rho_init(x)
        theta = theta_init(x)     
        return rho * (m/(2*np.pi*k*theta))**(1./2.) * np.exp(-m*v**2./(2.*k*theta))

def f(v, x, t):
    
    if(x<=0):
        theta = theta_walls(x)
        return 1 * (m/(2*np.pi*k*theta))**(1./2.) * np.exp(-m*v**2./(2.*k*theta))

    if(x>=1):
        theta = theta_walls(x)
        return 1 * (m/(2*np.pi*k*theta))**(1./2.) * np.exp(-m*v**2./(2.*k*theta))
                

    else:
        return f0(v, x - v*t)

def rho_integrand(v, x, t):
    integral_measure = 1
    return integral_measure * f(v, x, t)

def pressure_integrand(v, x, t):
    integral_measure = 1
    return integral_measure * v**2. * f(v, x, t)

rho_initial  = 1.0
temp_initial = 1.5
temp_walls   = 2.0

# Setting up spatial parameters:
N_x = np.array([32, 64, 128, 256, 512])     # No of spatial grid points

for i in range(N_x.size):
    
    print('Computing for Nx =', N_x[i])

    dx = 1./N_x[i]  # Step size for spatial grid
    x  = (np.arange(N_x[i]) + 0.5)*dx

    m = 1.0 # Mass of particle
    k = 1.0 # Boltzmann constant

    soln = np.zeros([N_x[i]])

    for grid_point in range(N_x[i]):

        integral1 = nquad(pressure_integrand, [[-np.inf, np.inf]], 
                          args=(x[grid_point], 2)
                         )
        integral2 = nquad(rho_integrand, [[-np.inf, np.inf]], 
                          args=(x[grid_point], 2)
                         )

        soln[grid_point] = integral1[0]/integral2[0]

    h5f = h5py.File('analytical_' + str(N_x[i]) + '.h5', 'w')
    h5f.create_dataset('temperature', data = soln)
    h5f.close()
