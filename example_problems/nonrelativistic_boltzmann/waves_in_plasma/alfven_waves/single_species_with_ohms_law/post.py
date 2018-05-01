import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as pl

import h5py
import domain
import params

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 30, 15 #10, 14
pl.rcParams['figure.dpi']      = 80
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

q1 = domain.q1_start + (np.arange(N_q1)) * dq1
q2 = domain.q2_start + (np.arange(N_q2)) * dq2

q2_l, q1_l = np.meshgrid(q2, q1)

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * dq1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * dq2

q2_c, q1_c = np.meshgrid(q2, q1)

# Taking user input:
# ndim_plotting   = input('1D or 2D plotting?:')
N_s             = int(input('Enter number of species: '))

# quantities      = ['density', 'v1', 'v2', 'v3', 'temperature', 'pressure', 'q1', 'q2', 'q3',
#                    'E1', 'E2', 'E3', 'B1', 'B2', 'B3'
#                   ]
# Taking input on quantities to be plotted:
# quantities    = input('Enter quantities to be plotted separated by commas:')
# quantities    = quantities.split(',')
# N_quantities  = len(quantities)
# N_rows        = input('Enter number of rows:')
# N_columns     = input('Enter number of columns:')

# ('Eigenvalue   = ', -9.68977236633914e-18 - 0.009442719099991581*I)
# (delta_u2_i, ' = ', 5.551115123125783e-16 + 0.6881909602355867*I)
# (delta_u3_i, ' = ', 0.6881909602355871)
# (delta_B2, ' = ', -3.469446951953614e-17 - 0.1624598481164533*I)
# (delta_B3, ' = ', -0.16245984811645306 + 1.3877787807814457e-16*I)

def B2_analytic(q1, t):
    
    omega = -9.68977236633914e-18 - 0.009442719099991581*1j

    B2_analytic = (params.amplitude * (-3.469446951953614e-17 - 0.1624598481164533 * 1j) * \
                   np.exp(  1j * params.k_q1 * q1
                         + omega * t
                        )).real

    return(B2_analytic)


def B3_analytic(q1, t):
    
    omega = -9.68977236633914e-18 - 0.009442719099991581*1j

    B3_analytic = params.amplitude * -0.16245984811645306 * \
                  np.exp(  1j * params.k_q1 * q1
                         + omega * t
                        ).real

    return(B3_analytic)

def v2_analytic(q1, t):
    
    omega = -9.68977236633914e-18 - 0.009442719099991581*1j

    v2_analytic = (params.amplitude * (5.551115123125783e-16 + 0.6881909602355867 * 1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + omega * t
                         )).real

    return(v2_analytic)

def v3_analytic(q1, t):
    
    omega = -9.68977236633914e-18 - 0.009442719099991581*1j

    v3_analytic = params.amplitude * 0.6881909602355871 * \
                  np.exp(  1j * params.k_q1 * q1
                         + omega * t
                        ).real

    return(v3_analytic)

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

#    heat_flux_1 = moments[:, :, 8*N_s:9*N_s] / n
#    heat_flux_2 = moments[:, :, 9*N_s:10*N_s] / n
#    heat_flux_3 = moments[:, :, 10*N_s:11*N_s] / n

    E1 = fields[:, :, 0]
    E2 = fields[:, :, 1]
    E3 = fields[:, :, 2]
    B1 = fields[:, :, 3]
    B2 = fields[:, :, 4]
    B3 = fields[:, :, 5]

    if(name == 'density'):
        return n

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
time_array = np.arange(0, 1 * params.t0 + params.dt_dump_moments, 
                       params.dt_dump_moments
                      )
indices = []

def determine_min_max(quantity):
    # Declaring an initial value for the max and minimum for the quantity plotted:
    q_max = -1e10 
    q_min = 1e10

    for time_index, t0 in enumerate(time_array):
    
        try:
            h5f  = h5py.File('dump_moments/t=%.3f'%(t0) + '.h5', 'r')
            moments = np.swapaxes(h5f['moments'][:], 0, 1)
            h5f.close()

            h5f    = h5py.File('dump_fields/t=%.3f'%(t0) + '.h5', 'r')
            fields = np.swapaxes(h5f['EM_fields'][:], 0, 1)
            h5f.close()

        except:
            indices.append(time_index)

        array = return_array_to_be_plotted(quantity, moments, fields)

        if(np.max(array)>q_max):
            q_max = np.max(array)

        if(np.min(array)<q_min):
            q_min = np.min(array)

    return(q_min, q_max)

# Traversal to determine the maximum and minimum:
time_array_post = time_array

B2_min, B2_max = determine_min_max('B2')
B3_min, B3_max = determine_min_max('B3')

v2_min, v2_max = determine_min_max('v2')
v3_min, v3_max = determine_min_max('v3')

time_array = np.delete(time_array, list(set(indices)))

def plot_1d():
    
    for time_index, t0 in enumerate(time_array):
        
        h5f  = h5py.File('dump_moments/t=%.3f'%(t0) + '.h5', 'r')
        moments = np.swapaxes(h5f['moments'][:], 0, 1)
        h5f.close()

        h5f    = h5py.File('dump_fields/t=%.3f'%(t0) + '.h5', 'r')
        fields = np.swapaxes(h5f['EM_fields'][:], 0, 1)
        h5f.close()

        v2 = return_array_to_be_plotted('v2', moments, fields)
        v3 = return_array_to_be_plotted('v3', moments, fields)

        B2 = return_array_to_be_plotted('B2', moments, fields)
        B3 = return_array_to_be_plotted('B3', moments, fields)

        fig = pl.figure()

        #ax1 = fig.add_subplot(2, 2, 1)
        #ax1.plot(q1[:, 0], delta_p[:, 0, 0], color = 'C0', label = 'Electrons')
        #ax1.plot(q1[:, 0], delta_p[:, 0, 1], '--', color = 'C3', label = 'Positrons')
        #ax1.legend()
        #ax1.set_xlabel(r'$x$')
        #ax1.set_ylabel(r'$\Delta p$')
        # ax1.set_ylim([0.98 * n_min, 1.02 * n_max])

        # ax2 = fig.add_subplot(2, 2, 2)
        # ax2.plot(q1[:, 0], v1[:, 0, 0], color = 'C0')
        # ax2.plot(q1[:, 0], v1[:, 0, 1], '--', color = 'C3')
        # ax2.set_xlabel(r'$x$')
        # ax2.set_ylabel(r'$v_x$')
        # ax2.set_ylim([0.98 * v1_min, 1.02 * v1_max])

        # ax3 = fig.add_subplot(2, 2, 3)
        # ax3.plot(q1[:, 0], p[:, 0, 0], color = 'C0')
        # ax3.plot(q1[:, 0], p[:, 0, 1], '--', color = 'C3')
        # ax3.set_xlabel(r'$x$')
        # ax3.set_ylabel(r'$p$')
        # ax3.set_ylim([0.98 * p_min, 1.02 * p_max])

        # ax4 = fig.add_subplot(3, 1, 1)
        # ax4.plot(q1[:, 0], np.sqrt(0*B1[:, 0]**2 + B2[:, 0]**2 + B3[:, 0]**2))
        # ax4.set_xlabel(r'$\frac{x}{l_s}$')
        # ax4.set_ylabel(r'$|B|$')
        # ax4.set_ylim([0, 1.02 * np.sqrt(0*B1_max**2 + B2_max**2 + B3_max**2)])

        # ax5 = fig.add_subplot(3, 1, 2)
        # ax5.plot(q1[:, 0], B2[:, 0])
        # ax5.set_xlabel(r'$\frac{x}{l_s}$')
        # ax5.set_ylabel(r'$B_y$')
        # ax5.set_ylim([1.02 * B2_min, 1.02 * B2_max])

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(q1_l[:, 0], B2[:, 0])
        ax1.plot(q1_l[:, 0], B2_analytic(q1_l[:, 0], t0) , '--', color = 'black')
        ax1.set_xlabel(r'$\frac{x}{l_s}$')
        ax1.set_ylabel(r'$B_y$')
        ax1.set_ylim([1.02 * B2_min, 1.02 * B2_max])

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(q1_l[:, 0], B3[:, 0])
        ax2.plot(q1_l[:, 0], B3_analytic(q1_l[:, 0], t0) , '--', color = 'black')
        ax2.set_xlabel(r'$\frac{x}{l_s}$')
        ax2.set_ylabel(r'$B_z$')
        ax2.set_ylim([1.02 * B3_min, 1.02 * B3_max])

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(q1_c[:, 0], v2[:, 0])
        ax3.plot(q1_c[:, 0], v2_analytic(q1_c[:, 0], t0) , '--', color = 'black')
        ax3.set_xlabel(r'$\frac{x}{l_s}$')
        ax3.set_ylabel(r'$v_y$')
        ax3.set_ylim([1.02 * v2_min, 1.02 * v2_max])

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(q1_c[:, 0], v3[:, 0])
        ax4.plot(q1_c[:, 0], v3_analytic(q1_c[:, 0], t0) , '--', color = 'black')
        ax4.set_xlabel(r'$\frac{x}{l_s}$')
        ax4.set_ylabel(r'$v_z$')
        ax4.set_ylim([1.02 * v3_min, 1.02 * v3_max])

        # fig.tight_layout()
        fig.suptitle('Time = %.3f'%(t0 / params.t0) + r' $\tau_A$')
        pl.savefig('images/%04d'%(time_index) + '.png')
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
        p  = return_array_to_be_plotted('pressure', moments, fields)
        B1 = return_array_to_be_plotted('B1', moments, fields)

        # n[:, :, 0] = -1 * n[:, :, 0] 
        pl.contourf(q1, q2, n[:, :, 1], np.linspace(n_min, n_max, 100))
        pl.xlabel(r'$\frac{x}{\lambda_D}$')
        pl.ylabel(r'$\frac{y}{\lambda_D}$')
        pl.gca().set_aspect('equal')
        pl.colorbar()
        pl.title('Time = %.2f'%(t0/params.t0) + r' $\omega_{cyclotron}^{-1}$')
        pl.savefig('images/%04d'%time_index + '.png')
        pl.clf()

plot_1d()
