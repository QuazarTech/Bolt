#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np

from .utils.fft_funcs import fft2, ifft2
from .utils.broadcasted_primitive_operations import multiply

def df_hat_dt_multimode_evolution(f_hat, self):
    """
    Returns the value of the derivative of the f_hat with respect to time 
    respect to time. This is used to evolve the system in time.

    Input:
    ------

      f_hat  : Fourier mode values for the distribution function at which the slope is computed
               At t = 0 the initial state of the system is passed to this function:

    Output:
    -------
    df_dt : The time-derivative of f_hat
    """
    (A_q1, A_q2) = self._A_q(f_hat, self.time_elapsed, 
                             self.q1_center, self.q2_center,
                             self.p1, self.p2, self.p3,
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
                                        self.p1, self.p2, self.p3,
                                        self.compute_moments, 
                                        self.physical_system.params
                                       )
                          )/(self.N_q2 * self.N_q1)

        df_hat_dt += C_f_hat

    if(self.physical_system.params.EM_fields_enabled == True):
        
        if(self.physical_system.params.fields_type == 'electrostatic'):
            rho = multiply(self.physical_system.params.charge,
                           self.compute_moments('density', f_hat=f_hat)
                          )
            self.fields_solver.compute_electrostatic_fields(rho)

        else:
            J1 = multiply(self.physical_system.params.charge,
                          self.compute_moments('mom_v1_bulk', f_hat=f_hat)
                         ) 
            J2 = multiply(self.physical_system.params.charge,
                          self.compute_moments('mom_v2_bulk', f_hat=f_hat)
                         ) 
            J3 = multiply(self.physical_system.params.charge,
                          self.compute_moments('mom_v3_bulk', f_hat=f_hat)
                         ) 

            self.fields_solver.evolve_electrodynamic_fields(J1, J2, J3)
        
        # get_fields for linear solver returns the mode amplitudes of the fields
        # So, we obtain A_p1_hat, A_p2_hat, A_p3_hat
        (A_p1_hat, A_p2_hat, A_p3_hat) = af.broadcast(self._A_p, f_hat, self.time_elapsed,
                                                      self.q1_center, self.q2_center,
                                                      self.p1, self.p2, self.p3,
                                                      self.fields_solver, self.physical_system.params
                                                     )

        fields_term =   multiply(A_p1_hat, self.dfdp1_background) \
                      + multiply(A_p2_hat, self.dfdp2_background) \
                      + multiply(A_p3_hat, self.dfdp3_background)

        # print(af.sum(af.abs(fields_term)))
        # print(af.sum(af.abs(df_hat_dt)))
        df_hat_dt -= fields_term

    af.eval(df_hat_dt)
    return(df_hat_dt)

def df_hat_dt(f_hat, self):

    return(df_hat_dt_multimode_evolution(f_hat, self))

    # if(self.single_mode_evolution == True):
    #     return(dY_dt_singlemode_evolution(Y, self))

    # else:
    #     return(dY_dt_multimode_evolution(Y, self))
