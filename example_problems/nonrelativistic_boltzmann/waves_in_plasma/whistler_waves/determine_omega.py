import numpy as np
from scipy.optimize import curve_fit

import h5py
import domain
import params

N_q1 = domain.N_q1

dq1 = (domain.q1_end - domain.q1_start) / N_q1
q1  = domain.q1_start + (np.arange(N_q1)) * dq1

def E3_analytic(q1, omega_i):
    
    omega_r = 0
    E3_analytic = (params.amplitude * (2.1076890233118206e-16 - 0.3747499256299707*1j) * \
                   np.exp(  1j * params.k_q1 * q1
                          + (omega_r - 1j * abs(omega_i)) * params.t_final
                         )).real

    return(E3_analytic)

h5f = h5py.File('dump_3/N_%04d'%(N_q1) + '.h5')
E3  = h5f['EM_fields'][:][0, :, 2]
h5f.close()

popt, pcov = curve_fit(E3_analytic, q1 + dq1 / 2, E3)
print(popt)
