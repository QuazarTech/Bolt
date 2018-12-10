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

# Taking user input:
# ndim_plotting   = input('1D or 2D plotting?:')
N_s             = 2 #int(input('Enter number of species: '))

# quantities      = ['density', 'v1', 'v2', 'v3', 'temperature', 'pressure', 'q1', 'q2', 'q3',
#                    'E1', 'E2', 'E3', 'B1', 'B2', 'B3'
#                   ]
# Taking input on quantities to be plotted:
# quantities    = input('Enter quantities to be plotted separated by commas:')
# quantities    = quantities.split(',')
# N_quantities  = len(quantities)
# N_rows        = input('Enter number of rows:')
# N_columns     = input('Enter number of columns:')

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

# Declaration of the time array:
# time_array = np.arange(0, 300 * params.t0 + params.dt_dump_moments, 
#                        params.dt_dump_moments
#                       )

# Traversal to determine the maximum and minimum:
def determine_min_max(quantity):
    # Declaring an initial value for the max and minimum for the quantity plotted:
    q_max = 0    
    q_min = 1e10

    for time_index, t0 in enumerate(time_array):
        
        h5f  = h5py.File('dump_moments/t=%.3f'%(t0) + '.h5', 'r')
        moments = np.swapaxes(h5f['moments'][:], 0, 1)
        h5f.close()

        h5f    = h5py.File('dump_fields/t=%.3f'%(t0) + '.h5', 'r')
        fields = np.swapaxes(h5f['EM_fields'][:], 0, 1)
        h5f.close()

        array = return_array_to_be_plotted(quantity, moments, fields)

        if(np.max(array)>q_max):
            q_max = np.max(array)

        if(np.min(array)<q_min):
            q_min = np.min(array)

    return(q_min, q_max)

# n_min, n_max   = determine_min_max('density')
# v1_min, v1_max = determine_min_max('v1')
# T_min, T_max   = determine_min_max('temperature')
# B1_min, B1_max = determine_min_max('B1')

def plot_1d():
    for time_index, t0 in enumerate(time_array):
        
        h5f  = h5py.File('dump_moments/t=%.3f'%(t0) + '.h5', 'r')
        moments = np.swapaxes(h5f['moments'][:], 0, 1)
        h5f.close()

        h5f    = h5py.File('dump_fields/t=%.3f'%(t0) + '.h5', 'r')
        fields = np.swapaxes(h5f['EM_fields'][:], 0, 1)
        h5f.close()

        n  = return_array_to_be_plotted('density', moments, fields)
        v1 = return_array_to_be_plotted('v1', moments, fields)
        T  = return_array_to_be_plotted('temperature', moments, fields)
        B1 = return_array_to_be_plotted('B1', moments, fields)

        fig = pl.figure()
  
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(q2[0, :], n[0, :, 0], color = 'C0', label = 'Electrons')
        ax1.plot(q2[0, :], n[0, :, 1], '--', color = 'C3', label = 'Ions')
        ax1.legend()
        ax1.set_xlabel(r'$y(l_s)$')
        ax1.set_ylabel(r'$n(n_0)$')
        ax1.set_ylim([0.95 * n_min, 1.05 * n_max])

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(q2[0, :], v1[0, :, 0] / params.v0, color = 'C0')
        ax2.plot(q2[0, :], v1[0, :, 1] / params.v0, '--', color = 'C3')
        ax2.set_xlabel(r'$y(l_s)$')
        ax2.set_ylabel(r'$v_x(v_0)$')
        ax2.set_ylim([1.05 * v1_min / params.v0, 1.05 * v1_max / params.v0])

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(q2[0, :], T[0, :, 0] / params.T0, color = 'C0')
        ax3.plot(q2[0, :], T[0, :, 1] / params.T0, '--', color = 'C3')
        ax3.set_xlabel(r'$y(l_s)$')
        ax3.set_ylabel(r'$T(T_0)$')
        ax3.set_ylim([0.95 * T_min / params.T0, 1.05 * T_max / params.T0])

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(q2[0, :], B1[0, :] / params.B0)
        ax4.set_xlabel(r'$y(l_s)$')
        ax4.set_ylabel(r'$B_x(\sqrt{n_0 m_0} v_0)$')
        ax4.set_ylim([0.95 * B1_min / params.B0, 1.05 * B1_max / params.B0])

        # fig.tight_layout()
        fig.suptitle('Time = %.2f'%(t0 / params.t0)+r'$\omega_c^{-1}$=%.2f'%(t0 * params.plasma_frequency)+r'$\omega_p^{-1}$')
        pl.savefig('images/%04d'%time_index + '.png')
        pl.close(fig)
        pl.clf()

def plot_2d():

    for time_index, t0 in enumerate(time_array):

        h5f  = h5py.File('dump_moments/t=%.3f'%(t0) + '.h5', 'r')
        moments = np.swapaxes(h5f['moments'][:], 0, 1)
        h5f.close()

        h5f    = h5py.File('dump_fields/t=%.3f'%(t0) + '.h5', 'r')
        fields = np.swapaxes(h5f['EM_fields'][:], 0, 1)
        h5f.close()

        n  = return_array_to_be_plotted('density', moments, fields)
        v1 = return_array_to_be_plotted('v1', moments, fields)
        T  = return_array_to_be_plotted('temperature', moments, fields)
        B1 = return_array_to_be_plotted('B1', moments, fields)

        fig = pl.figure()

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.contourf(q1, q2, n[:, :, 0], np.linspace(0.99 * n_min, 1.01 * n_max, 100))
        ax1.set_title(r'$n(n_0)$')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.contourf(q1, q2, v1[:, :, 0] / params.v0, np.linspace(0.99 * v1_min / params.v0, 1.01 * v1_max / params.v0, 100))
        ax2.set_title(r'$v_x(v_0)$')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.contourf(q1, q2, T[:, :, 0] / params.T0, np.linspace(0.99 * T_min / params.T0, 1.01 * T_max / params.T0, 100))
        ax3.set_title(r'$T(T_0)$')

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.contourf(q1, q2, B1[:, :, 0] / params.B0, np.linspace(0.99 * B1_min / params.B0, 1.01 * B1_max / params.B0, 100))
        ax4.set_title(r'$B_x(\sqrt{n_0 m_0} v_0)$')

        # fig.tight_layout()
        fig.suptitle('Time = %.2f'%(t0 / params.t0)+r'$\omega_c^{-1}$=%.2f'%(t0 * params.plasma_frequency)+r'$\omega_p^{-1}$')
        pl.savefig('images/%04d'%time_index + '.png')
        pl.close(fig)
        pl.clf()

# plot_1d()

h5f      = h5py.File('fields_data_initial.h5', 'r')
fields_i = np.swapaxes(h5f['EM_fields'][:], 0, 1)
h5f.close()

h5f      = h5py.File('fields_data_final.h5', 'r')
fields_f = np.swapaxes(h5f['EM_fields'][:], 0, 1)
h5f.close()

pl.plot(q2[0, :], (fields_i[:, :, 5])[0, :] / params.B0, color = 'C2', label = r'$t = 0$')
pl.plot(q2[0, :], (fields_f[:, :, 5])[0, :] / params.B0, color = 'C1', label = r'$t = 1000 \omega_c^{-1}$')
pl.legend(framealpha = 0, fontsize = 20, bbox_to_anchor = (0.4, 0.69))
pl.xlabel(r'$y(l_s)$')
pl.ylabel(r'$B_x(B_0)$')
pl.tight_layout()
pl.savefig('plot.png', bbox_inches = 'tight')

# h5f       = h5py.File('moments_data_initial.h5', 'r')
# moments_i = np.swapaxes(h5f['moments'][:], 0, 1)
# h5f.close()

# h5f       = h5py.File('moments_data_final.h5', 'r')
# moments_f = np.swapaxes(h5f['moments'][:], 0, 1)
# h5f.close()

# pl.plot(q2[0, :], (return_array_to_be_plotted('density', moments_f, moments_f)[:, :, 0])[0, :] / params.n0, label = r'Electrons')
# pl.plot(q2[0, :], (return_array_to_be_plotted('density', moments_f, moments_f)[:, :, 1])[0, :] / params.n0, '--', color = 'C3', label = r'Ions')
# # pl.legend(framealpha = 0)
# pl.xlabel(r'$y(l_s)$')
# pl.ylabel(r'$n(n_0)$')
# pl.savefig('plot.png', bbox_inches = 'tight')
