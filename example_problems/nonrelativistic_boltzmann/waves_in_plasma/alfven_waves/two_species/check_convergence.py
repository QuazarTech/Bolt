import numpy as np
import h5py
import matplotlib as mpl 
mpl.use('agg')
import pylab as pl

import domain as domain
import params as params

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20
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

# ('Eigenvalue   = ', 5.3944386867730924e-17 - 0.0898800439758432*I)
# (delta_u2_e, ' = ', -4.85722573273506e-15 - 0.333061857862197*I)
# (delta_u3_e, ' = ', -0.333061857862222 - 3.885780586188048e-16*I)
# (delta_u2_i, ' = ', -4.801714581503802e-15 - 0.3692429960259134*I)
# (delta_u3_i, ' = ', -0.3692429960259359 + 1.8041124150158794e-16*I)
# (delta_B2, ' = ', 5.6066262743570405e-15 + 0.37389325198333345*I)
# (delta_B3, ' = ', 0.37389325198336115)
# (delta_E2, ' = ', 0.336055419305355 + 4.996003610813204e-16*I)
# (delta_E3, ' = ', -4.7878367936959876e-15 - 0.33605541930533006*I)

def v2e_analytic(q1, t):
    
    omega = 5.3944386867730924e-17 - 0.0898800439758432 * 1j

    u2e_analytic = (params.amplitude * (-4.85722573273506e-15 - 0.333061857862197*1j) * \
                    np.exp(  1j * params.k_q1 * q1
                           + omega * t
                          )).real

    return(u2e_analytic)

def v3e_analytic(q1, t):
    
    omega = 5.3944386867730924e-17 - 0.0898800439758432 * 1j

    u3e_analytic = (params.amplitude * -0.333061857862222 * \
                   np.exp(  1j * params.k_q1 * q1
                          + omega * t
                         )).real

    return(u3e_analytic)

def v2i_analytic(q1, t):
    
    omega = 5.3944386867730924e-17 - 0.0898800439758432 * 1j

    u2i_analytic = (params.amplitude * (-4.801714581503802e-15 - 0.3692429960259134*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + omega * t
                         )).real

    return(u2i_analytic)

def v3i_analytic(q1, t):
    
    omega = 5.3944386867730924e-17 - 0.0898800439758432 * 1j

    u3i_analytic = (params.amplitude * -0.3692429960259359 * \
                   np.exp(  1j * params.k_q1 * q1
                          + omega * t
                         )).real

    return(u3i_analytic)

def E2_analytic(q1, t):
    
    omega = 5.3944386867730924e-17 - 0.0898800439758432 * 1j

    E2_analytic = (params.amplitude * (0.336055419305355 + 4.996003610813204e-16*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + omega * t
                         )).real

    return(E2_analytic)

def E3_analytic(q1, t):
    
    omega = 5.3944386867730924e-17 - 0.0898800439758432 * 1j

    E3_analytic = (params.amplitude * (-4.7878367936959876e-15 - 0.33605541930533006*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + omega * t
                         )).real

    return(E3_analytic)

def B2_analytic(q1, t):
    
    omega = 5.3944386867730924e-17 - 0.0898800439758432 * 1j

    B2_analytic = (params.amplitude * (5.6066262743570405e-15 + 0.37389325198333345*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + omega * t
                         )).real

    return(B2_analytic)

def B3_analytic(q1, t):
    
    omega = 5.3944386867730924e-17 - 0.0898800439758432 * 1j

    B3_analytic = (params.amplitude * 0.37389325198336115 * \
                   np.exp(  1j * params.k_q1 * q1
                          + omega * t
                         )).real

    return(B3_analytic)

N = np.array([32, 48, 64, 96, 128])

error_v2e = np.zeros(N.size)
error_v3e = np.zeros(N.size)
error_v2i = np.zeros(N.size)
error_v3i = np.zeros(N.size)

error_E2 = np.zeros(N.size)
error_E3 = np.zeros(N.size)
error_B2 = np.zeros(N.size)
error_B3 = np.zeros(N.size)

for i in range(N.size):

    dq1 = (domain.q1_end - domain.q1_start) / int(N[i])
    q1  = domain.q1_start + (np.arange(int(N[i]))) * dq1

    # Timestep as set by the CFL condition:
    dt_fvm = params.N_cfl * dq1 \
                          / max(domain.p1_end + domain.p2_end + domain.p3_end) # joining elements of the list

    dt_fdtd = params.N_cfl * dq1 \
                           / params.c # lightspeed

    dt        = min(dt_fvm, dt_fdtd)

    h5f = h5py.File('dump_1/N_%04d'%(int(N[i])) + '.h5')
    n_e = h5f['moments'][:][0, :, 0]
    n_i = h5f['moments'][:][0, :, 1]
    v_2e = h5f['moments'][:][0, :, 12] / n_e
    v_2i = h5f['moments'][:][0, :, 13] / n_i
    v_3e = h5f['moments'][:][0, :, 14] / n_e
    v_3i = h5f['moments'][:][0, :, 15] / n_i
    h5f.close()

    h5f = h5py.File('dump_2/N_%04d'%(int(N[i])) + '.h5')
    E2  = h5f['EM_fields'][:][0, :, 1]
    E3  = h5f['EM_fields'][:][0, :, 2]
    B2  = h5f['EM_fields'][:][0, :, 4]
    B3  = h5f['EM_fields'][:][0, :, 5]
    h5f.close()

    v2e_ana = v2e_analytic(q1 + dq1 / 2, params.t_final)
    v2i_ana = v2i_analytic(q1 + dq1 / 2, params.t_final)
    v3e_ana = v3e_analytic(q1 + dq1 / 2, params.t_final)
    v3i_ana = v3i_analytic(q1 + dq1 / 2, params.t_final)

    E2_ana = E2_analytic(q1 + dq1 / 2, params.t_final)
    E3_ana = E3_analytic(q1 + dq1 / 2, params.t_final)
    B2_ana = B2_analytic(q1, params.t_final + dt / 2)
    B3_ana = B3_analytic(q1, params.t_final + dt / 2)

    error_v2e[i] = np.mean(abs(v_2e - v2e_ana))
    error_v2i[i] = np.mean(abs(v_2i - v2i_ana))
    error_v3e[i] = np.mean(abs(v_3e - v3e_ana))
    error_v3i[i] = np.mean(abs(v_3i - v3i_ana))

    error_E2[i] = np.mean(abs(E2 - E2_ana))
    error_E3[i] = np.mean(abs(E3 - E3_ana))
    error_B2[i] = np.mean(abs(B2 - B2_ana))
    error_B3[i] = np.mean(abs(B3 - B3_ana))

print('Errors Obtained:')
print('L1 norm of error for v2e:', error_v2e)
print('L1 norm of error for v2i:', error_v2i)
print('L1 norm of error for v3e:', error_v3e)
print('L1 norm of error for v3i:', error_v3i)
print()
print('L1 norm of error for E2:', error_E2)
print('L1 norm of error for E3:', error_E3)
print('L1 norm of error for B2:', error_B2)
print('L1 norm of error for B3:', error_B3)

print('\nConvergence Rates:')
print('Order of convergence for v2e:', np.polyfit(np.log10(N), np.log10(error_v2e), 1)[0])
print('Order of convergence for v2i:', np.polyfit(np.log10(N), np.log10(error_v2i), 1)[0])
print('Order of convergence for v3e:', np.polyfit(np.log10(N), np.log10(error_v3e), 1)[0])
print('Order of convergence for v3i:', np.polyfit(np.log10(N), np.log10(error_v3i), 1)[0])
print()
print('Order of convergence for E2:', np.polyfit(np.log10(N), np.log10(error_E2), 1)[0])
print('Order of convergence for E3:', np.polyfit(np.log10(N), np.log10(error_E3), 1)[0])
print('Order of convergence for B2:', np.polyfit(np.log10(N), np.log10(error_B2), 1)[0])
print('Order of convergence for B3:', np.polyfit(np.log10(N), np.log10(error_B3), 1)[0])

# pl.loglog(N, error_n, '-o', label = 'Density')
# pl.loglog(N, error_v1, '-o', label = 'Velocity')
# pl.loglog(N, error_T, '-o', label = 'Temperature')
# pl.loglog(N, error_n[0]*32**2/N**2, '--', color = 'black', label = r'$O(N^{-2})$')
# pl.xlabel(r'$N$')
# pl.ylabel('Error')
# pl.legend()
# pl.savefig('convergenceplot.png')
