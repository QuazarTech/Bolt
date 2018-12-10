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
# ('Eigenvalue   = ', 9.71525577391238e-17 - 0.09749003842237096*I)
def k_zero_point_one(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (-5.578870698741412e-15 + 0.3579577660636656*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)


# ('Eigenvalue   = ', -1.6653344672133062e-16 - 0.31564181352535126*I)
def k_zero_point_three(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (-1.0524486013588352e-15 - 0.37543492280290663*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', 6.66133719674266e-16 - 0.5642357758698509*I)
def k_zero_point_five(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (8.819923048213943e-16 - 0.38824299607278534*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -1.1102230395306195e-16 - 0.8413143822113407*I)
def k_zero_point_seven(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (-3.20966194703072e-17 - 0.3965643888135162*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -4.4408917643978883e-16 - 1.1435239413243987*I)
def k_zero_point_nine(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (0.40083580763931453 + 2.0159591932422221e-16*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -1.3322672363149143e-15 - 1.4665159765326323*I)
def k_one_point_one(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (-6.297775989619429e-16 - 0.40159465386855214*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', 4.440903875955698e-16 - 1.8054050770752403*I)
def k_one_point_three(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (4.440892098500626e-16 - 0.3993794338412208*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -1.1102241320796426e-15 - 2.1551822180280933*I)
def k_one_point_five(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (-1.7867651802561113e-16 - 0.39468140164821736*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -1.3402211154212169e-21 - 2.511030696097157*I)
def k_one_point_seven(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (0.38793007322916484 - 3.7470027081099033e-16*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -1.2212550643605751e-15 - 2.868535104378185*I)
def k_one_point_nine(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (0.3794960354426654 - 2.246466901389965e-16*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -4.441500142749952e-16 - 3.2237988180301373*I)
def k_two_point_one(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (0.3697001126191448 + 3.9963692077815693e-16*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -4.4312378386199074e-16 - 3.745370158177388*I)
def k_two_point_three(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (1.3104404375323918e-16 - 0.35305801902921286*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -1.778678888935013e-15 - 3.9148649135154923*I)
def k_two_point_five(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (1.8537358402307726e-16 - 0.3471137402123639*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', -1.7754382345917728e-15 - 4.245706179714867*I)
def k_two_point_seven(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (3.2896747968107296e-17 - 0.3347940463083292*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

# ('Eigenvalue   = ', 1.5541928563873292e-15 - 4.564320146512589*I)
def k_two_point_nine(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (0.3220627278651147 - 1.3877787807814457e-16*1j) * \
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
