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

N_s = 1

def return_array_to_be_plotted(name, moments, fields):
    m       = np.array(params.mass).reshape(1, 1, len(params.mass))
    n       = moments[:, :, 0:N_s]
    
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

    E1 = fields[:, :, 0]
    E2 = fields[:, :, 1]
    E3 = fields[:, :, 2]
    B1 = fields[:, :, 3]
    B2 = fields[:, :, 4]
    B3 = fields[:, :, 5]

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

    elif(name == 'q1'):
        return heat_flux_1

    elif(name == 'q2'):
        return heat_flux_2

    elif(name == 'q3'):
        return heat_flux_3

    elif(name == 'E1'):
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

# Time parameters:
dt      = 0.001
t_final = params.t_final

time_array  = np.arange(dt, t_final + dt, dt)

kinetic_energy_error = np.zeros(time_array.size)
mass_error           = np.zeros(time_array.size)
momentum_x_error     = np.zeros(time_array.size)
momentum_y_error     = np.zeros(time_array.size)

kinetic_energy_initial = 0
mass_initial           = 0
momentum_x_initial     = 0
momentum_y_initial     = 0

for time_index, t0 in enumerate(time_array):

    h5f  = h5py.File('dump_moments/t=%.3f'%(t0) + '.h5', 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    h5f   = h5py.File('dump_moments2/t=%.3f'%(t0) + '.h5', 'r')
    moments2 = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    E  = return_array_to_be_plotted('energy', moments, moments)
    n  = return_array_to_be_plotted('density', moments, moments)
    n2 = return_array_to_be_plotted('density', moments2, moments2)
    px = return_array_to_be_plotted('v1', moments, moments) * return_array_to_be_plotted('density', moments, moments) 
    py = return_array_to_be_plotted('v2', moments, moments) * return_array_to_be_plotted('density', moments, moments) 
    
    if(time_index == 0):
        print(n.shape)
        mass_initial           = np.mean(n)
        momentum_x_initial     = np.mean(px)
        momentum_y_initial     = np.mean(py)
        kinetic_energy_initial = np.mean(E)

    mass           = np.max(n)
    mass2          = np.max(n2)
    momentum_x     = np.mean(px)
    momentum_y     = np.mean(py)
    kinetic_energy = np.mean(E) 

    mass_error[time_index]           = (mass - 0*mass_initial)
    momentum_x_error[time_index]     = (mass2 - 0 * momentum_x - 0 * momentum_x_initial)
    momentum_y_error[time_index]     = (momentum_y - momentum_y_initial)
    kinetic_energy_error[time_index] = (kinetic_energy - kinetic_energy_initial)

pl.semilogy(time_array, abs(mass_error), label = r'$N=32$', alpha = 1)
pl.semilogy(time_array, abs(momentum_x_error), label = r'$N=64$', alpha = 1)
# pl.semilogy(time_array, abs(momentum_y_error), label = 'y-Momentum', alpha = 0.5)
# pl.semilogy(time_array, abs(kinetic_energy_error), label = 'Kinetic Energy', alpha = 0.5)
# pl.ylim([1e-17, 1e-11])
pl.ylabel(r'MAX($n$)')
pl.legend(framealpha = 0, fontsize = 25)
pl.xlabel(r'Time')
pl.savefig('plot.png', bbox_inches = 'tight')
