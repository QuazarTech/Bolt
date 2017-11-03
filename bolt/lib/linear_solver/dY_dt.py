#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import arrayfire as af
import numpy as np

from .EM_fields_solver import compute_electrostatic_fields

def dY_dt_multimode_evolution(Y, self):
    """
    Returns the value of the derivative of the fourier mode quantities 
    of the distribution function, and the field quantities with 
    respect to time. This is used to evolve the system in time.

    Input:
    ------

      Y  : The array Y is the state of the system as given by the result of 
           the last time-step's integration. The elements of Y, hold the 
           following data:
     
           f_hat   = Y[0]
           E1_hat = Y[1]
           E2_hat = Y[2]
           E3_hat = Y[3]
           B1_hat = Y[4]
           B2_hat = Y[5]
           B3_hat = Y[6]
     
           At t = 0 the initial state of the system is passed to this function:

    Output:
    -------
    dY_dt : The time-derivatives of all the quantities stored in Y
    """
    f_hat = Y[:, :, :, 0]
    
    self.E1_hat = Y[:, :, :, 1]
    self.E2_hat = Y[:, :, :, 2]
    self.E3_hat = Y[:, :, :, 3]
    self.B1_hat = Y[:, :, :, 4]
    self.B2_hat = Y[:, :, :, 5]
    self.B3_hat = Y[:, :, :, 6]

    # Scaling Appropriately:
    f       = af.real(af.ifft2(0.5 * self.N_q2 * self.N_q1 * f_hat))
    
    C_f_hat = 2 * af.fft2(self._source(f, self.q1_center, self.q2_center,
                                       self.p1, self.p2, self.p3,
                                       self.compute_moments, 
                                       self.physical_system.params
                                      )
                         )/(self.N_q2 * self.N_q1)

    if(self.physical_system.params.fields_solver == 'electrostatic' or
       self.physical_system.params.fields_solver == 'fft'):
        compute_electrostatic_fields(self)

    elif(self.physical_system.params.fields_solver == 'fdtd'):
        pass

    else:
        raise NotImplementedError('Method invalid/not-implemented')
    
    mom_bulk_p1 = self.compute_moments('mom_p1_bulk')
    mom_bulk_p2 = self.compute_moments('mom_p2_bulk')
    mom_bulk_p3 = self.compute_moments('mom_p3_bulk')

    J1_hat = 2 * af.fft2(  self.physical_system.params.charge_electron 
                         * mom_bulk_p1
                         )/(self.N_q1 * self.N_q2)
    
    J2_hat = 2 * af.fft2(  self.physical_system.params.charge_electron
                         * mom_bulk_p2
                        )/(self.N_q1 * self.N_q2)

    J3_hat = 2 * af.fft2(  self.physical_system.params.charge_electron
                         * mom_bulk_p3
                         )/(self.N_q1 * self.N_q2)

    # Defining lambda functions to perform broadcasting operations:
    # This is done using af.broadcast, which allows us to perform 
    # batched operations when operating on arrays of different sizes:

    multiply = lambda a,b:a * b
    addition = lambda a,b:a + b
    
    # af.broadcast(function, *args) performs batched operations on
    # function(*args):

    dE1_hat_dt = af.broadcast(addition, 
                              af.broadcast(multiply, self.B3_hat, 1j * self.k_q2),
                              - J1_hat
                             )

    dE2_hat_dt = af.broadcast(addition,
                              af.broadcast(multiply,-self.B3_hat, 1j * self.k_q1),
                              - J2_hat
                             )

    dE3_hat_dt = af.broadcast(addition, 
                                af.broadcast(multiply, self.B2_hat, 1j * self.k_q1)
                              - af.broadcast(multiply, self.B1_hat, 1j * self.k_q2), 
                              - J3_hat
                             )

    dB1_hat_dt = af.broadcast(multiply, -self.E3_hat, 1j * self.k_q2)
    dB2_hat_dt = af.broadcast(multiply, self.E3_hat, 1j * self.k_q1)
    dB3_hat_dt =   af.broadcast(multiply, self.E1_hat, 1j * self.k_q2) \
                 - af.broadcast(multiply, self.E2_hat, 1j * self.k_q1)

    (A_p1, A_p2, A_p3) = af.broadcast(self._A_p, self.q1_center, self.q2_center,
                                      self.p1, self.p2, self.p3,
                                      self.E1_hat, self.E2_hat, self.E3_hat,
                                      self.B1_hat, self.B2_hat, self.B3_hat,
                                      self.physical_system.params
                                     )

    df_hat_dt  = -1j * (  af.broadcast(multiply, self.k_q1, self._A_q1)
                        + af.broadcast(multiply, self.k_q2, self._A_q2)
                       ) * f_hat

    
    # Adding the fields term only when charge is non-zero
    if(self.physical_system.params.charge_electron != 0):
        fields_term =   af.broadcast(multiply, A_p1, self.dfdp1_background) \
                      + af.broadcast(multiply, A_p2, self.dfdp2_background) \
                      + af.broadcast(multiply, A_p3, self.dfdp3_background)
        df_hat_dt  -= fields_term

    # Avoiding addition of the fields term when tau != inf
    tau = self.physical_system.params.tau(self.q1_center, 
                                          self.q2_center,
                                          self.p1, self.p2, self.p3
                                         )

    df_hat_dt += C_f_hat

    df_hat_dt += af.select(self.physical_system.params.tau(self.q1_center, 
                                                           self.q2_center,
                                                           self.p1, self.p2, self.p3
                                                          ) != np.inf,\
                           C_f_hat,\
                           0
                          )
    
    # Obtaining the dY_dt vector by joining the derivative quantities of
    # the individual distribution function and field modes:
    dY_dt = af.join(3, af.join(3, df_hat_dt, dE1_hat_dt, dE2_hat_dt, dE3_hat_dt),
                    dB1_hat_dt, dB2_hat_dt, dB3_hat_dt
                   )

    af.eval(dY_dt)
    return(dY_dt)

def dY_dt_singlemode_evolution(Y, self):

    k_q1 = self.physical_system.params.k_q1   
    k_q2 = self.physical_system.params.k_q2

    charge_electron = self.physical_system.params.charge_electron

    delta_f_hat  = Y[0]
    delta_E1_hat = Y[1]
    delta_E2_hat = Y[2]
    delta_E3_hat = Y[3]
    
    delta_B1_hat = Y[4]
    delta_B2_hat = Y[5]
    delta_B3_hat = Y[6]

    if(self.physical_system.params.fields_solver == 'electrostatic' or
       self.physical_system.params.fields_solver == 'fft'):
        compute_electrostatic_fields(self)
        delta_E1_hat = self.delta_E1_hat
        delta_E2_hat = self.delta_E1_hat

    elif(self.physical_system.params.fields_solver == 'fdtd'):
        pass

    else:
        raise NotImplementedError('Method invalid/not-implemented')

    delta_p1_bulk =   self.compute_moments('mom_p1_bulk') \
                    / self.physical_system.params.rho_background
    delta_p2_bulk =   self.compute_moments('mom_p2_bulk') \
                    / self.physical_system.params.rho_background
    delta_p3_bulk =   self.compute_moments('mom_p3_bulk') \
                    / self.physical_system.params.rho_background

    delta_J1_hat = charge_electron * delta_p1_bulk
    delta_J2_hat = charge_electron * delta_p2_bulk
    delta_J3_hat = charge_electron * delta_p3_bulk

    ddelta_E1_hat_dt = (delta_B3_hat * 1j * k_q2) - delta_J1_hat
    ddelta_E2_hat_dt = (- delta_B3_hat * 1j * k_q1) - delta_J2_hat
    ddelta_E3_hat_dt = (delta_B2_hat * 1j * k_q1 - delta_B1_hat * 1j * k_q1) - delta_J3_hat

    ddelta_B1_hat_dt = (- delta_E3_hat * 1j * k_q2)
    ddelta_B2_hat_dt = (delta_E3_hat * 1j * k_q1)
    ddelta_B3_hat_dt = (delta_E1_hat * 1j * k_q2 - delta_E2_hat * 1j * k_q1)

    fields_term =   charge_electron * (  delta_E1_hat \
                                       + delta_B3_hat * self.p2 \
                                       - delta_B2_hat * self.p3
                                      ) * self.dfdp1_background \
                  + charge_electron * (  delta_E2_hat \
                                       + delta_B1_hat * self.p3 \
                                       - delta_B3_hat * self.p1
                                      ) * self.dfdp2_background \
                  + charge_electron * (  delta_E3_hat \
                                       + delta_B2_hat * self.p1 \
                                       - delta_B1_hat * self.p2
                                      ) * self.dfdp3_background

    C_f_hat = self._source(delta_f_hat,
                           self.p1, self.p2, self.p3,
                           self.compute_moments, 
                           self.physical_system.params
                          )

    ddelta_f_hat_dt = - 1j * (k_q1 * self.p1 + k_q2 * self.p2) * delta_f_hat \
                      - fields_term + C_f_hat
  
    dY_dt = np.array([ddelta_f_hat_dt,
                      ddelta_E1_hat_dt, ddelta_E1_hat_dt, ddelta_E3_hat_dt,
                      ddelta_B1_hat_dt, ddelta_B3_hat_dt, ddelta_B3_hat_dt
                     ]
                    )
  
    return(dY_dt)

def dY_dt(Y, self):

    if(self.single_mode_evolution == True):
        return(dY_dt_singlemode_evolution(Y, self))

    else:
        return(dY_dt_multimode_evolution(Y, self))
