#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np

from bolt.lib.utils.fft_funcs import fft2, ifft2
from bolt.lib.utils.broadcasted_primitive_operations import multiply

def df_hat_dt(f_hat, fields_hat, self):
    """
    Returns the value of the derivative of the f_hat with respect to time 
    respect to time. This is used to evolve the system in time.

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
    (A_q1, A_q2) = self._A_q(self.time_elapsed, 
                             self.q1_center, self.q2_center,
                             self.p1_center, self.p2_center, self.p3_center,
                             self.physical_system.params
                            )

    df_hat_dt = -1j * (  multiply(self.k_q1, A_q1)
                       + multiply(self.k_q2, A_q2)
                      ) * f_hat

    if(self.physical_system.params.source_enabled == True):
        
        # Scaling Appropriately:
        f = af.real(ifft2(0.5 * self.N_q2 * self.N_q1 * 
                          f_hat
                         )
                   )

        C_f_hat = 2 * fft2(self._source(f, self.time_elapsed, 
                                        self.q1_center, self.q2_center,
                                        self.p1_center, self.p2_center, self.p3_center,
                                        self.compute_moments, 
                                        self.physical_system.params
                                       )
                          )/(self.N_q2 * self.N_q1)

        df_hat_dt += C_f_hat

    if(self.physical_system.params.fields_enabled == True):
        
        if(self.physical_system.params.fields_type == 'electrostatic'):
            
            rho_hat = multiply(self.physical_system.params.charge,
                               self.compute_moments('density', f_hat=f_hat)
                              )
            self.fields_solver.compute_electrostatic_fields(rho_hat)

        elif(self.physical_system.params.fields_type == 'electrodynamic'):
            # Handled by dfields_hat_dt
            pass

        # Used in debugging; advection tests for p-space where fields are to be held constant 
        elif(self.physical_system.params.fields_type == 'None'):
            pass

        else:
            raise NotImplementedError('Invalid option for fields solver!')

        # get_fields for linear solver returns the mode amplitudes of the fields
        # So, we obtain A_p1_hat, A_p2_hat, A_p3_hat
        (A_p1_hat, A_p2_hat, A_p3_hat) = af.broadcast(self._A_p, self.time_elapsed,
                                                      self.q1_center, self.q2_center,
                                                      self.p1_center, self.p2_center, self.p3_center,
                                                      self.fields_solver, self.physical_system.params
                                                     )

        fields_term =   multiply(A_p1_hat, self.dfdp1_background) \
                      + multiply(A_p2_hat, self.dfdp2_background) \
                      + multiply(A_p3_hat, self.dfdp3_background)

        # Including the mean magnetic field term:
        # Multiplying by q_center**0 to get the values in the array of required dimension
        # Dividing by 2 to normalize appropriately(look at the initialization sector):
        B1_mean = af.mean(self.fields_solver.fields_hat[3, 0, 0, 0]) * self.q1_center**0 / 2
        B2_mean = af.mean(self.fields_solver.fields_hat[4, 0, 0, 0]) * self.q1_center**0 / 2
        B3_mean = af.mean(self.fields_solver.fields_hat[5, 0, 0, 0]) * self.q1_center**0 / 2

        e = self.physical_system.params.charge
        m = self.physical_system.params.mass

        # Converting delta_f array to velocity_expanded form:
        f_hat_p_expanded = af.moddims(f_hat,
                                      self.N_p1, self.N_p2, self.N_p3,
                                      self.N_species * self.N_q1 * self.N_q2
                                     )

        # Computing ddelta_f_dp using a 4th order finite different stencil:
        ddelta_f_dp1 = multiply((-af.shift(f_hat_p_expanded, -2) + 8 * af.shift(f_hat_p_expanded, -1)
                                 +af.shift(f_hat_p_expanded,  2) - 8 * af.shift(f_hat_p_expanded,  1)
                                ), 1 / (12 * self.dp1)
                               )

        ddelta_f_dp2 = multiply((-af.shift(f_hat_p_expanded, 0, -2) + 8 * af.shift(f_hat_p_expanded, 0, -1)
                                 +af.shift(f_hat_p_expanded, 0,  2) - 8 * af.shift(f_hat_p_expanded, 0,  1)
                                ), 1 / (12 * self.dp2)
                               )

        ddelta_f_dp3 = multiply((-af.shift(f_hat_p_expanded, 0, 0, -2) + 8 * af.shift(f_hat_p_expanded, 0, 0, -1)
                                 +af.shift(f_hat_p_expanded, 0, 0,  2) - 8 * af.shift(f_hat_p_expanded, 0, 0,  1)
                                ), 1 / (12 * self.dp3)
                               )

        # Converting back to positions expanded:
        ddelta_f_dp1 = af.moddims(ddelta_f_dp1,
                                  self.N_p1 * self.N_p2 * self.N_p3,
                                  self.N_species, self.N_q1, self.N_q2
                                 )

        ddelta_f_dp2 = af.moddims(ddelta_f_dp2,
                                  self.N_p1 * self.N_p2 * self.N_p3,
                                  self.N_species, self.N_q1, self.N_q2
                                 )

        ddelta_f_dp3 = af.moddims(ddelta_f_dp3,
                                  self.N_p1 * self.N_p2 * self.N_p3,
                                  self.N_species, self.N_q1, self.N_q2
                                 )

        fields_term_mean_magnetic_fields = \
            multiply(e/m, (  (multiply(self.p2_center, B3_mean) - multiply(self.p3_center, B2_mean)) * ddelta_f_dp1
                           + (multiply(self.p3_center, B1_mean) - multiply(self.p1_center, B3_mean)) * ddelta_f_dp2
                           + (multiply(self.p1_center, B2_mean) - multiply(self.p2_center, B1_mean)) * ddelta_f_dp3
                          )
                    )

        df_hat_dt -= fields_term + fields_term_mean_magnetic_fields

    af.eval(df_hat_dt)
    return(df_hat_dt)
