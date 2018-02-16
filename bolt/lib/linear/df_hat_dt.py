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
    (A_q1, A_q2) = self._A_q(f_hat, self.time_elapsed, 
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
        (A_p1_hat, A_p2_hat, A_p3_hat) = af.broadcast(self._A_p, f_hat, self.time_elapsed,
                                                      self.q1_center, self.q2_center,
                                                      self.p1_center, self.p2_center, self.p3_center,
                                                      self.fields_solver, self.physical_system.params
                                                     )

        fields_term =   multiply(A_p1_hat, self.dfdp1_background) \
                      + multiply(A_p2_hat, self.dfdp2_background) \
                      + multiply(A_p3_hat, self.dfdp3_background)

        df_hat_dt -= fields_term

    af.eval(df_hat_dt)
    return(df_hat_dt)
