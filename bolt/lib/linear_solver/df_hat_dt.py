#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import arrayfire as af
import numpy as np

from .EM_fields_solver import compute_electrostatic_fields

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
    # Getting the fields:
    if(   self.physical_system.params.fields_solver == 'electrostatic'
       or self.physical_system.params.fields_solver == 'fft'
      ):
        compute_electrostatic_fields(self, f_hat=f_hat)

    elif(self.physical_system.params.fields_solver == 'fdtd'):
        
        for i in range(self.N_species):
            mom_bulk_p1 = self.compute_moments('mom_v1_bulk', i, f_hat=f_hat)
            mom_bulk_p2 = self.compute_moments('mom_v2_bulk', i, f_hat=f_hat)
            mom_bulk_p3 = self.compute_moments('mom_v3_bulk', i, f_hat=f_hat)

            if(i == 0):
                self.J1_hat = 2 * af.fft2(  self.physical_system.params.charge[i] 
                                          * mom_bulk_p1
                                         )/(self.N_q1 * self.N_q2)
                
                self.J2_hat = 2 * af.fft2(  self.physical_system.params.charge[i]
                                     * mom_bulk_p2
                                    )/(self.N_q1 * self.N_q2)

                self.J3_hat = 2 * af.fft2(  self.physical_system.params.charge[i]
                                     * mom_bulk_p3
                                    )/(self.N_q1 * self.N_q2)

            else:
                self.J1_hat += 2 * af.fft2(  self.physical_system.params.charge[i] 
                                      * mom_bulk_p1
                                     )/(self.N_q1 * self.N_q2)
                
                self.J2_hat += 2 * af.fft2(  self.physical_system.params.charge[i]
                                      * mom_bulk_p2
                                     )/(self.N_q1 * self.N_q2)

                self.J3_hat += 2 * af.fft2(  self.physical_system.params.charge[i]
                                      * mom_bulk_p3
                                     )/(self.N_q1 * self.N_q2)

    else:
        raise NotImplementedError('Method invalid/not-implemented')

    for i in range(self.N_species):
        A_q1 = self._A_q(self.q1_center, self.q2_center,
                         self.p1, self.p2, self.p3,
                         self.physical_system.params, i
                        )[0]

        A_q2 = self._A_q(self.q1_center, self.q2_center,
                         self.p1, self.p2, self.p3,
                         self.physical_system.params, i
                        )[1]

        # Scaling Appropriately:
        f       = af.real(af.ifft2(0.5 * self.N_q2 * self.N_q1 * 
                                   self.f_hat[:, :, i * dof:(i+1) * dof]
                                  )
                         )

        C_f_hat = 2 * af.fft2(self._source(f, self.q1_center, self.q2_center,
                                           self.p1, self.p2, self.p3,
                                           self.compute_moments, 
                                           self.physical_system.params, i
                                          )
                             )/(self.N_q2 * self.N_q1)

        (A_p1, A_p2, A_p3) = af.broadcast(self._A_p, self.q1_center, self.q2_center,
                                          self.p1, self.p2, self.p3,
                                          self.E1_hat, self.E2_hat, self.E3_hat,
                                          self.B1_hat, self.B2_hat, self.B3_hat,
                                          self.physical_system.params, i
                                         )

        multiply = lambda a,b:a*b

        fields_term =   af.broadcast(multiply, A_p1, self.dfdp1_background[i]) \
                      + af.broadcast(multiply, A_p2, self.dfdp2_background[i]) \
                      + af.broadcast(multiply, A_p3, self.dfdp3_background[i])

        self.df_hat_dt[:, :, i * dof:(i+1) * dof] = \
            -1j * (  af.broadcast(multiply, self.k_q1, A_q1)
                   + af.broadcast(multiply, self.k_q2, A_q2)
                  ) * f_hat[:, :, i * dof:(i+1) * dof] + C_f_hat - fields_term
    
    af.eval(self.df_hat_dt)
    return(self.df_hat_dt)

def df_hat_dt(f_hat, self):

    return(df_hat_dt_multimode_evolution(f_hat, self))

    # if(self.single_mode_evolution == True):
    #     return(dY_dt_singlemode_evolution(Y, self))

    # else:
    #     return(dY_dt_multimode_evolution(Y, self))
