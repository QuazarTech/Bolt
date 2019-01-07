import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as pl

import h5py
import domain
import params

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 9, 4
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 30
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'

N_q1 = domain.N_q1
N_q2 = domain.N_q2
N_g  = domain.N_ghost

dq1 = (domain.q1_end - domain.q1_start) / N_q1
dq2 = (domain.q2_end - domain.q2_start) / N_q2

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * dq1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * dq2

q2, q1 = np.meshgrid(q2, q1)

# Finding the number of species, by looking at the number of elements in mass:
N_s = len(params.mass)

def return_moment_to_be_plotted(name, moments):
    """
    Returns the quantity of interest to the user. This is provided
    by giving a string indicating quantity to plot. All these quantities
    returned are at cell centers. Allowed inputs:
    
    'density' 
    'energy', 
    'v1', 'v2', 'v3' --> bulk velocities
    'J1', 'J2', 'J3' --> currents
    'temperature'
    'p1', 'p2', 'p3' --> 'p_x', 'p_y', 'p_z'
    'pressure'       --> total pressure
    
    'heat_flux_x', 
    'heat_flux_y', --> heat fluxes
    'heat_flux_z' 
    
    NOTE: This function returns the quantity of interest for all species. For
          instance, if we have two species, return_moment_to_be_plotted('density', moments)
          would return an array of shape(N_q1, N_q2, 2), where array[:, :, 0] denotes
          the first species and array[:, :, 1] denotes the second species 

    Parameters
    ----------

    name : str
           Pass the name of the quantity that needs to be plotted.

    moments: np.ndarray
             This is the array containing the data that is contained in the
             file that is written by dump_moments. NOTE: This function expects
             this array in the format (N_q1, N_q2, N_s).
    """

    m = np.array(params.mass).reshape(1, 1, N_s)
    e = np.array(params.charge).reshape(1, 1, N_s)
    n = moments[:, :, 0:N_s]
    
    v1_bulk = moments[:, :, 5*N_s:6*N_s] / n
    v2_bulk = moments[:, :, 6*N_s:7*N_s] / n
    v3_bulk = moments[:, :, 7*N_s:8*N_s] / n
    
    p1 = m * (  2 * moments[:, :, 2*N_s:3*N_s]
              - n * v1_bulk**2
             )

    p2 = m * (  2 * moments[:, :, 3*N_s:4*N_s]
              - n * v2_bulk**2
             )

    p3 = m * (  2 * moments[:, :, 4*N_s:5*N_s]
              - n * v3_bulk**2
             )

    T       = m * (  2 * moments[:, :, 1*N_s:2*N_s]
                   - n * v1_bulk**2
                   - n * v2_bulk**2
                   - n * v3_bulk**2
                  ) / (params.p_dim * n)

    heat_flux_1 = moments[:, :, 8*N_s:9*N_s] / n
    heat_flux_2 = moments[:, :, 9*N_s:10*N_s] / n
    heat_flux_3 = moments[:, :, 10*N_s:11*N_s] / n

    if(name == 'density'):
        return n

    elif(name == 'energy'):
        return m * moments[:, :, 1*N_s:2*N_s]

    elif(name == 'v1'):
        return v1_bulk

    elif(name == 'v2'):
        return v2_bulk

    elif(name == 'v3'):
        return v3_bulk

    elif(name == 'J1'):
        # Summing along axis 2, to sum across all species
        return np.sum(n * e * v1_bulk, 2)

    elif(name == 'J2'):
        # Summing along axis 2, to sum across all species
        return np.sum(n * e * v2_bulk, 2)

    elif(name == 'J3'):
        # Summing along axis 2, to sum across all species
        return np.sum(n * e * v3_bulk, 2)

    elif(name == 'temperature'):
        return T

    elif(name == 'p1'):
        return(p1)

    elif(name == 'p2'):
        return(p2)

    elif(name == 'p3'):
        return(p3)

    elif(name == 'pressure'):
        return(n * T)

    elif(name == 'heat_flux_x'):
        return heat_flux_1

    elif(name == 'heat_flux_y'):
        return heat_flux_2

    elif(name == 'heat_flux_z'):
        return heat_flux_3

    else:
        raise Exception('Not valid!')

def return_field_to_be_plotted(name, fields):
    """
    Returns the field of interest to the user. This is provided
    by giving a string indicating field to plot. All these quantities
    returned are on the grid locations:

    B1 --> (i + 1/2, j)
    B2 --> (i, j + 1/2)
    B3 --> (i, j)

    E1 --> (i, j + 1/2)
    E2 --> (i + 1/2, j)
    E3 --> (i + 1/2, j + 1/2)

    Allowed inputs: 'E1', 'E2', 'E3', 'B1', 'B2', 'B3;

    Parameters
    ----------

    name : str
           Pass the name of the field quantity that needs to be plotted.

    moments: np.ndarray
             This is the array containing the data that is contained in the
             file that is written by dump_EM_fields. NOTE: This function expects
             this array in the format (N_q1, N_q2, N_s).
    """

    E1 = fields[:, :, 0]
    E2 = fields[:, :, 1]
    E3 = fields[:, :, 2]
    B1 = fields[:, :, 3]
    B2 = fields[:, :, 4]
    B3 = fields[:, :, 5]

    if(name == 'E1'):
        return E1

    elif(name == 'E2'):
        return E2

    elif(name == 'E3'):
        return E3

    elif(name == 'B1'):
        return B1

    elif(name == 'B2'):
        return B2

    elif(name == 'B3'):
        return B3

    else:
        raise Exception('Not valid!')

# Traversal to determine the maximum and minimum:
def determine_min_max(name, time_array):
    """
    Determines the minimum and maximum of the quantity of interest over
    the entire time_array. This is needed when we want to make movies showing
    evolution in time. By using this function we avoid the problem of shifting
    limits in y axis / changing colorbar limits.

    Parameters
    ----------

    name : str
           Pass the name of the  quantity that needs to be plotted.

    time_array: np.ndarray
                The time array over which the maximum and minimum of the
                quantity need to be determined.
    """
    # Declaring an initial value for the max and minimum for the quantity plotted:
    q_max = 0    
    q_min = 1e10

    for time_index, t0 in enumerate(time_array):
        
        if(name in ['E1', 'E2', 'E3', 'B1', 'B2', 'B3']):
            h5f    = h5py.File('dump_fields/t=%.3f'%(t0) + '.h5', 'r')
            fields = np.swapaxes(h5f['EM_fields'][:], 0, 1)
            h5f.close()

            array = return_field_to_be_plotted(name, fields)

        else:
            h5f  = h5py.File('dump_moments/t=%.3f'%(t0) + '.h5', 'r')
            moments = np.swapaxes(h5f['moments'][:], 0, 1)
            h5f.close()

            array = return_moment_to_be_plotted(name, moments)

        if(np.max(array)>q_max):
            q_max = np.max(array)

        if(np.min(array)<q_min):
            q_min = np.min(array)

    return(q_min, q_max)
