#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

from bolt.lib.utils.fft_funcs import fft2, ifft2
from bolt.lib.utils.broadcasted_primitive_operations import multiply

def dfields_hat_dt(f_hat, fields_hat, self):
    """
    Returns the value of the derivative of the fields_hat with respect to time 
    respect to time. This is used to evolve the fields with time. 
    
    NOTE:All the fields quantities are included in fields_hat as follows:

    E1_hat = fields_hat[0]
    E2_hat = fields_hat[1]
    E3_hat = fields_hat[2]

    B1_hat = fields_hat[3]
    B2_hat = fields_hat[4]
    B3_hat = fields_hat[5]

    Input:
    ------

      f_hat  : Fourier mode values for the distribution function at which the slope is computed
               At t = 0 the initial state of the system is passed to this function:

      fields_hat  : Fourier mode values for the fields at which the slope is computed
                    At t = 0 the initial state of the system is passed to this function:

    Output:
    -------
    df_dt : The time-derivative of f_hat
    """
    eps = self.physical_system.params.eps
    mu  = self.physical_system.params.mu

    B1_hat = fields_hat[3]
    B2_hat = fields_hat[4]
    B3_hat = fields_hat[5]

    if(self.physical_system.params.hybrid_model_enabled == True):
        
        # curlB_x =  dB3/dq2
        curlB_1 = B3_hat * 1j * self.k_q2
        # curlB_y = -dB3/dq1
        curlB_2 = -B3_hat * 1j * self.k_q1
        # curlB_z = (dB2/dq1 - dB1/dq2)
        curlB_3 = (B2_hat * 1j * self.k_q1 - B1_hat * 1j * self.k_q2)
    
        # c --> inf limit: J = (∇ x B) / μ
        J1_hat = curlB_1 / mu 
        J2_hat = curlB_2 / mu
        J3_hat = curlB_3 / mu

    else:
        
        J1_hat = multiply(self.physical_system.params.charge,
                          self.compute_moments('mom_v1_bulk', f_hat=f_hat)
                         ) 
        J2_hat = multiply(self.physical_system.params.charge,
                          self.compute_moments('mom_v2_bulk', f_hat=f_hat)
                         ) 
        J3_hat = multiply(self.physical_system.params.charge,
                          self.compute_moments('mom_v3_bulk', f_hat=f_hat)
                         ) 

    # Summing along all species:
    J1_hat = af.sum(J1_hat, 1)
    J2_hat = af.sum(J2_hat, 1)
    J3_hat = af.sum(J3_hat, 1)

    # Checking that there is no mean field component:
    # try:
    #     assert(af.mean(af.abs(B1_hat[:, 0, 0])) < 1e-12)
    #     assert(af.mean(af.abs(B2_hat[:, 0, 0])) < 1e-12)
    #     assert(af.mean(af.abs(B3_hat[:, 0, 0])) < 1e-12)
    # except:
    #     raise SystemExit('Linear Solver cannot solve for non-zero mean magnetic fields')

    # Equations Solved:
    # dE1/dt = + dB3/dq2 - J1
    # dE2/dt = - dB3/dq1 - J2
    # dE3/dt = dB2/dq1 - dB1/dq2 - J3

    if(self.physical_system.params.hybrid_model_enabled == True):

        # Using Generalized Ohm's Law for electric field:
        n_i_hat = self.compute_moments('density', f_hat=f_hat)

        n_i = af.real(ifft2(0.5 * self.N_q2 * self.N_q1 * n_i_hat))
        v1  = af.real(ifft2(0.5 * self.N_q2 * self.N_q1 * self.compute_moments('mom_v1_bulk', f_hat=f_hat))) / n_i
        v2  = af.real(ifft2(0.5 * self.N_q2 * self.N_q1 * self.compute_moments('mom_v2_bulk', f_hat=f_hat))) / n_i
        v3  = af.real(ifft2(0.5 * self.N_q2 * self.N_q1 * self.compute_moments('mom_v3_bulk', f_hat=f_hat))) / n_i

        B1 = af.real(ifft2(0.5 * self.N_q2 * self.N_q1 * B1_hat))
        B2 = af.real(ifft2(0.5 * self.N_q2 * self.N_q1 * B2_hat))
        B3 = af.real(ifft2(0.5 * self.N_q2 * self.N_q1 * B3_hat))

        J1 = af.real(ifft2(0.5 * self.N_q2 * self.N_q1 * J1_hat))
        J2 = af.real(ifft2(0.5 * self.N_q2 * self.N_q1 * J2_hat))
        J3 = af.real(ifft2(0.5 * self.N_q2 * self.N_q1 * J3_hat))

        T_e = self.physical_system.params.fluid_electron_temperature
        # Computing the quantities needed and then doing FT again:
        # # (v X B)_x = B3 * v2 - B2 * v3
        # v_cross_B_1 = B3 * v2 - B2 * v3
        # # (v X B)_y = B1 * v3 - B3 * v1
        # v_cross_B_2 = B1 * v3 - B3 * v1
        # # (v X B)_z = B2 * v1 - B1 * v2
        # v_cross_B_3 = B2 * v1 - B1 * v2

        # # (J X B)_x = B3 * J2 - B2 * J3
        # J_cross_B_1 = B3 * J2 - B2 * J3
        # # (J X B)_y = B1 * J3 - B3 * J1
        # J_cross_B_2 = B1 * J3 - B3 * J1
        # # (J X B)_z = B2 * J1 - B1 * J2
        # J_cross_B_3 = B2 * J1 - B1 * J2

        # # Using a 4th order stencil:
        # dn_q1 = (-     af.shift(n_i, 0, 0, -2) + 8 * af.shift(n_i, 0, 0, -1) 
        #          - 8 * af.shift(n_i, 0, 0,  1) +     af.shift(n_i, 0, 0,  2)
        #         ) / (12 * self.dq1)

        # dn_q2 = (-     af.shift(n_i, 0, 0, 0, -2) + 8 * af.shift(n_i, 0, 0, 0, -1) 
        #          - 8 * af.shift(n_i, 0, 0, 0,  1) +     af.shift(n_i, 0, 0, 0,  2)
        #         ) / (12 * self.dq2)

        # # E = -(v X B) + (J X B) / (en) - T ∇n / (en)
        # E1 = - v_cross_B_1 + J_cross_B_1 / multiply(self.physical_system.params.charge, n_i) \
        #      - T_e * dn_q1 / multiply(self.physical_system.params.charge, n_i)

        # E2 = - v_cross_B_2 + J_cross_B_2 / multiply(self.physical_system.params.charge, n_i) \
        #      - T_e * dn_q2 / multiply(self.physical_system.params.charge, n_i)

        # E3 = - v_cross_B_3 + J_cross_B_3 / multiply(self.physical_system.params.charge, n_i) 

        # E1_hat = 2 * fft2(E1) / (self.N_q1 * self.N_q2)
        # E2_hat = 2 * fft2(E2) / (self.N_q1 * self.N_q2) - T_e * n_i_hat * self.k_q2 / af.mean(n_i)
        # E3_hat = 2 * fft2(E3) / (self.N_q1 * self.N_q2)

        # background magnetic fields == 0 ALWAYS:
        # (v X B)_x = B3 * v2 - B2 * v3
        v_cross_B_1_hat = B3_hat * af.mean(v2) - B2_hat * af.mean(v3)
        # (v X B)_y = B1 * v3 - B3 * v1
        v_cross_B_2_hat = B1_hat * af.mean(v3) - B3_hat * af.mean(v1)
        # (v X B)_z = B2 * v1 - B1 * v2
        v_cross_B_3_hat = B2_hat * af.mean(v1) - B1_hat * af.mean(v2)

        # (J X B)_x = B3 * J2 - B2 * J3
        J_cross_B_1_hat = B3_hat * af.mean(J2) - B2_hat * af.mean(J3)
        # (J X B)_y = B1 * J3 - B3 * J1
        J_cross_B_2_hat = B1_hat * af.mean(J3) - B3_hat * af.mean(J1)
        # (J X B)_z = B2 * J1 - B1 * J2
        J_cross_B_3_hat = B2_hat * af.mean(J1) - B1_hat * af.mean(J2)

        E1_hat = - v_cross_B_1_hat + multiply(J_cross_B_1_hat, 1 / self.physical_system.params.charge) / af.mean(n_i) \
                 - T_e * n_i_hat * self.k_q1 / af.mean(n_i) 

        E2_hat = - v_cross_B_2_hat + multiply(J_cross_B_2_hat, 1 / self.physical_system.params.charge) / af.mean(n_i) \
                 - T_e * n_i_hat * self.k_q2 / af.mean(n_i) 

        E3_hat = - v_cross_B_3_hat + multiply(J_cross_B_3_hat, 1 / self.physical_system.params.charge) / af.mean(n_i)

        self.fields_solver.fields_hat[0] = E1_hat 
        self.fields_solver.fields_hat[1] = E2_hat 
        self.fields_solver.fields_hat[2] = E3_hat 

    else:
    
        E1_hat = fields_hat[0]
        E2_hat = fields_hat[1]
        E3_hat = fields_hat[2]
        
    dE1_hat_dt =  B3_hat * 1j * self.k_q2 / (mu * eps) - J1_hat / eps
    dE2_hat_dt = -B3_hat * 1j * self.k_q1 / (mu * eps) - J2_hat / eps
    dE3_hat_dt =  (B2_hat * 1j * self.k_q1 - B1_hat * 1j * self.k_q2) / (mu * eps) \
                 -J3_hat / eps

    # dB1/dt = - dE3/dq2
    # dB2/dt = + dE3/dq1
    # dB3/dt = - (dE2/dq1 - dE1/dq2)

    dB1_hat_dt = -E3_hat * 1j * self.k_q2
    dB2_hat_dt =  E3_hat * 1j * self.k_q1
    dB3_hat_dt =  E1_hat * 1j * self.k_q2 - E2_hat * 1j * self.k_q1

    dfields_hat_dt = af.join(0, 
                             af.join(0, dE1_hat_dt, dE2_hat_dt, dE3_hat_dt),
                             dB1_hat_dt, dB2_hat_dt, dB3_hat_dt
                            )

    af.eval(dfields_hat_dt)
    return(dfields_hat_dt)
