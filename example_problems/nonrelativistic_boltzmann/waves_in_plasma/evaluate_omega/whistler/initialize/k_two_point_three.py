"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

# Problem Parameters:
# n0_num  = 1
# B10_num = 1
# c_num   = 5
# mu_num  = 1
# e_num   = 1
# mi_num  = 1
# me_num  = 1 / 10
# L1_num  = 2 / 2.3 * pi
# k1_num  = 2 * pi / L1_num

# ('Eigenvalue   = ', 1.7761042622468226e-15 - 3.573493784028671*I)
# (delta_u2_e, ' = ', 0.5583486611611699)
# (delta_u3_e, ' = ', -3.278525725642495e-17 + 0.5583486611611693*I)
# (delta_u2_i, ' = ', 0.07845711202586898 - 1.6813815761234675e-17*I)
# (delta_u3_i, ' = ', 2.6172894553684253e-17 + 0.07845711202586896*I)
# (delta_B2, ' = ', -0.23094853732887538 - 3.0831999280056532e-18*I)
# (delta_B3, ' = ', 1.999844788467403e-18 - 0.2309485373288758*I)
# (delta_E2, ' = ', -1.4362883576706995e-16 - 0.35882311416315243*I)
# (delta_E3, ' = ', 0.35882311416315205 - 1.9448808205200716e-16*I)

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    k = params.boltzmann_constant

    v1_bulk   = 0

    # Assigning separate bulk velocities
    v2_bulk_i =   params.amplitude * 0.07845711202586898 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0 * af.sin(params.k_q1 * q1)

    v2_bulk_e =   params.amplitude * 0.5583486611611699 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0 * af.sin(params.k_q1 * q1)
    
    v3_bulk_i =   params.amplitude * 0 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0.07845711202586896 * af.sin(params.k_q1 * q1)

    v3_bulk_e =   params.amplitude * 0 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0.5583486611611693 * af.sin(params.k_q1 * q1)

    n = n_b + 0 * q1**0

    f_e = n * (m[0, 0] / (2 * np.pi * k * T_b)) \
            * af.exp(-m[0, 0] * (v2[:, 0] - v2_bulk_e)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 0] * (v3[:, 0] - v3_bulk_e)**2 / (2 * k * T_b))

    f_i = n * (m[0, 1] / (2 * np.pi * k * T_b)) \
            * af.exp(-m[0, 1] * (v2[:, 1] - v2_bulk_i)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 1] * (v3[:, 1] - v3_bulk_i)**2 / (2 * k * T_b))

    f = af.join(1, f_e, f_i)

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1**0
    
    E2 =   params.amplitude * 0 * af.cos(params.k_q1 * q1) \
         - params.amplitude * -0.35882311416315243 * af.sin(params.k_q1 * q1)
    
    E3 =   params.amplitude * 0.35882311416315205 * af.cos(params.k_q1 * q1) \
         - params.amplitude * 0 * af.sin(params.k_q1 * q1)

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    dt = params.dt
    B1 = params.B0 * q1**0

    omega = 1.7761042622468226e-15 - 3.573493784028671 * 1j

    B2 = (params.amplitude * (-0.23094853732887538 - 3.0831999280056532e-18*1j) * \
          np.exp(  1j * params.k_q1 * np.array(q1)
                 + omega * dt / 2
                )).real

    B3 = (params.amplitude * (1.999844788467403e-18 - 0.2309485373288758*1j) * \
          np.exp(  1j * params.k_q1 * np.array(q1)
                 + omega * dt / 2
                )).real

    B2 = af.moddims(af.to_array(B2), 1, 1, q1.shape[2], q1.shape[3])
    B3 = af.moddims(af.to_array(B3), 1, 1, q1.shape[2], q1.shape[3])

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
