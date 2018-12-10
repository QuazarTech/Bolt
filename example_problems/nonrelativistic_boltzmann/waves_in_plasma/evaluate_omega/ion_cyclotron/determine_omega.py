import numpy as np
from scipy.optimize import curve_fit

import matplotlib as mpl
mpl.use('agg')
import pylab as pl

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 15, 10
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

import h5py
import domain
import params.k_zero_point_one as params

N_q1 = domain.N_q1

dq1 = (domain.q1_end - domain.q1_start) / N_q1
q1  = domain.q1_start + (np.arange(N_q1)) * dq1

# We will be comparing E3 throughout:
# ('Eigenvalue   = ', 5.3944386867730924e-17 - 0.0898800439758432*I)
def k_zero_point_one(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (-4.7878367936959876e-15 - 0.33605541930533006*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)


# ('Eigenvalue   = ', -2.445960100389528e-16 - 0.24770193840644444*I)
def k_zero_point_three(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (-1.6930901125533637e-15 + 0.3105045761916173*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', 2.181414783511122e-16 - 0.378473560197061*I)
def k_zero_point_five(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (9.159339953157541e-16 + 0.28256288313081546*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -1.0386656874380749e-16 - 0.4854853296792793*I)
def k_zero_point_seven(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (2.636779683484747e-16 + 0.25373860465358244*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -3.3840660338578e-16 - 0.57232851937906*I)
def k_zero_point_nine(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (-4.8385010106448195e-17 + 0.2254761533604362*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -4.660033205834214e-17 - 0.6424994975912932*I)
def k_one_point_one(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (-2.6107588313450947e-16 + 0.19890202265731322*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', 1.5960424030073204e-16 - 0.6991495234104763*I)
def k_one_point_three(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (0.17471769676500315 - 2.185751579730777e-16*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -8.164138624051161e-17 - 0.7449709077916147*I)
def k_one_point_five(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (-1.7867651802561113e-16 + 0.15322895762201585*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -2.145963578826118e-16 - 0.7821809289122083*I)
def k_one_point_seven(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (5.551115123125783e-17 + 0.1344473296943089*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -5.3757945175912085e-17 - 0.8125616518681449*I)
def k_one_point_nine(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (0.1182028475640971 + 1.2368036282628392e-16*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -7.403321437792102e-18 - 0.8375236525870918*I)
def k_two_point_one(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (0.10423537046121167 - 1.6479873021779667e-17*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -1.111917929357007e-16 - 0.8581739285999422*I)
def k_two_point_three(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (0.0922560965133902 - 4.981993685865618e-17*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', 1.0996322016622969e-18 - 0.8753780073148297*I)
def k_two_point_five(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (-4.076727223487585e-17 + 0.08198343373747803*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', 2.297103973785922e-18 - 0.8898124410947876*I)
def k_two_point_seven(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (1.3691546438536917e-17 + 0.07316088265835408*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -1.628807259424849e-19 - 0.902007210907606*I)
def k_two_point_nine(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (-1.734723475976807e-18 + 0.06556382361214708*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

h5f = h5py.File('EM_fields_data.h5')
E3  = h5f['EM_fields'][:][0, :, 2]
h5f.close()

popt, pcov = curve_fit(k_zero_point_one, q1 + dq1 / 2, E3)
print(popt, pcov)

pl.plot(q1 + dq1 / 2, E3)
pl.plot(q1 + dq1 / 2, k_zero_point_one(q1 + dq1 / 2, *popt), '--', color = 'black')
pl.savefig('plot.png')
