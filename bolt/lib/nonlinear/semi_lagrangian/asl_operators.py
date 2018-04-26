#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file defines methods which will be used when the 
advective semilagrangian method is to be used.

The operations of solving for the q-space, collisions and
p-space are defined independantly so that they maybe called 
appropriately by the time-splitting operator methods. 
"""

import arrayfire as af
from ..temporal_evolution import integrators
from .interpolation_routines import f_interp_2d, f_interp_p_3d
from bolt.lib.utils.broadcasted_primitive_operations import multiply
from bolt.lib.nonlinear.communicate import communicate_fields

# Advection in q-space:
def op_advect_q(self, dt):
    """
    Solves the following part of the equations specified:
    
    df/dt + A_q1 df/dq1 + A_q2 df/dq2 = 0

    Parameters
    ----------

    dt : double
         Time-step size to evolve the system
    """
    self._communicate_f()
    self._apply_bcs_f()
    f_interp_2d(self, dt)

    af.eval(self.f)
    return

# Used to solve the source term:
def op_solve_src(self, dt):
    """
    Evolves the source term of the equations specified:
    
    df/dt = source
    
    Parameters
    ----------

    dt : double
         Time-step size to evolve the system
    """

    if(self.performance_test_flag == True):
        tic = af.time()

    # Solving for tau = 0 systems:
    if(self.physical_system.params.instantaneous_collisions == True):
        self.f = self._source(self.f, self.time_elapsed,
                              self.q1_center, self.q2_center,
                              self.p1_center, self.p2_center, self.p3_center, 
                              self.compute_moments, 
                              self.physical_system.params, 
                              True
                             )

    if(    self.physical_system.params.source_enabled == True 
       and self.physical_system.params.instantaneous_collisions != True
      ):
        self.f = integrators.RK2(self._source, self.f, dt, 
                                 self.time_elapsed, 
                                 self.q1_center, self.q2_center,
                                 self.p1_center, self.p2_center, 
                                 self.p3_center, self.compute_moments, 
                                 self.physical_system.params
                                )
    
    af.eval(self.f)
    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_sourcets += toc - tic
    
    return

def op_fields(self, dt):
    """
    Evolves the following part of the equations specified:

    df/dt + A_p1 df/dp1 + A_p2 df/dp1 + A_p3 df/dp1 = 0
    
    Parameters
    ----------
    dt : double
         Time-step size to evolve the system
    """

    if(self.performance_test_flag == True):
        tic = af.time()
    
    if(self.physical_system.params.fields_solver == 'electrostatic'):
        rho = multiply(self.physical_system.params.charge,
                       self.compute_moments('density')
                      )
        self.fields_solver.compute_electrostatic_fields(rho)
    
    # Evolving fields:
    if(self.physical_system.params.fields_solver == 'fdtd'):
        
        if(self.physical_system.params.hybrid_model_enabled == True):

            communicate_fields(self.fields_solver, True)
            B1 = self.fields_solver.yee_grid_EM_fields[3] # (i + 1/2, j)
            B2 = self.fields_solver.yee_grid_EM_fields[4] # (i, j + 1/2)
            B3 = self.fields_solver.yee_grid_EM_fields[5] # (i, j)

            B1_plus_q2 = af.shift(B1, 0, 0, 0, -1)

            B2_plus_q1 = af.shift(B2, 0, 0, -1, 0)

            B3_plus_q1 = af.shift(B3, 0, 0, -1, 0)
            B3_plus_q2 = af.shift(B3, 0, 0, 0, -1)

            # curlB_x =  dB3/dq2
            curlB_1 =  (B3_plus_q2 - B3) / self.dq2 # (i, j + 1/2)
            # curlB_y = -dB3/dq1
            curlB_2 = -(B3_plus_q1 - B3) / self.dq1 # (i + 1/2, j)
            # curlB_z = (dB2/dq1 - dB1/dq2)
            curlB_3 =  (B2_plus_q1 - B2) / self.dq1 - (B1_plus_q2 - B1) / self.dq2 # (i + 1/2, j + 1/2)

            # c --> inf limit: J = (∇ x B) / μ
            mu = self.physical_system.params.mu
            J1 = curlB_1 / mu # (i, j + 1/2)
            J2 = curlB_2 / mu # (i + 1/2, j)
            J3 = curlB_3 / mu # (i + 1/2, j + 1/2)
            
            # Using Generalized Ohm's Law for electric field:
            # (v X B)_x = B3 * v2 - B2 * v3
            # (v X B)_x --> (i, j + 1/2)
            v_cross_B_1 =   0.5 * (B3_plus_q2 + B3) * self.compute_moments('mom_v2_bulk', f = 0.5 * (self.f + af.shift(self.f, 0, 0, 1))) \
                                                    / self.compute_moments('density', f = 0.5 * (self.f + af.shift(self.f, 0, 0, 1))) \
                          - B2                      * self.compute_moments('mom_v3_bulk', f = 0.5 * (self.f + af.shift(self.f, 0, 0, 1))) \
                                                    / self.compute_moments('density', f = 0.5 * (self.f + af.shift(self.f, 0, 0, 1)))
            
            # (v X B)_y = B1 * v3 - B3 * v1
            # (v X B)_y --> (i + 1/2, j)
            v_cross_B_2 =   B1                      * self.compute_moments('mom_v3_bulk', f = 0.5 * (self.f + af.shift(self.f, 0, 0, 0, 1))) \
                                                    / self.compute_moments('density', f = 0.5 * (self.f + af.shift(self.f, 0, 0, 0, 1))) \
                          - 0.5 * (B3_plus_q1 + B3) * self.compute_moments('mom_v1_bulk', f = 0.5 * (self.f + af.shift(self.f, 0, 0, 0, 1))) \
                                                    / self.compute_moments('density', f = 0.5 * (self.f + af.shift(self.f, 0, 0, 0, 1)))
            # (v X B)_z = B2 * v1 - B1 * v2
            # (v X B)_z --> (i + 1/2, j + 1/2)
            v_cross_B_3 =   0.5 * (B2_plus_q1 + B2) * self.compute_moments('mom_v1_bulk', f = self.f) \
                                                    / self.compute_moments('density', f = self.f) \
                          - 0.5 * (B1_plus_q2 + B1) * self.compute_moments('mom_v2_bulk', f = self.f) \
                                                    / self.compute_moments('density', f = self.f)

            # (J X B)_x = B3 * J2 - B2 * J3
            # (J X B)_x --> (i, j + 1/2)
            J_cross_B_1 =   0.5 * (B3_plus_q2 + B3) * (  J2 + af.shift(J2, 0, 0, 0, -1)
                                                       + af.shift(J2, 0, 0, 1) + af.shift(J2, 0, 0, 1, -1)
                                                      ) * 0.25 \
                          - B2                      * (af.shift(J3, 0, 0, 1) + J3) * 0.5

            # (J X B)_y = B1 * J3 - B3 * J1
            # (J X B)_y --> (i + 1/2, j)
            J_cross_B_2 =   B1                      * (af.shift(J3, 0, 0, 0, 1) + J3) * 0.5 \
                          - 0.5 * (B3_plus_q1 + B3) * (  J1 + af.shift(J1, 0, 0, 0, 1)
                                                       + af.shift(J1, 0, 0, -1) + af.shift(J1, 0, 0, -1, 1)
                                                      ) * 0.25

            # (J X B)_z = B2 * J1 - B1 * J2
            # (J X B)_z --> (i + 1/2, j + 1/2)
            J_cross_B_3 =   0.5 * (B2_plus_q1 + B2) * (af.shift(J1, 0, 0, -1) + J1) * 0.5 \
                          - 0.5 * (B1_plus_q2 + B1) * (af.shift(J2, 0, 0, 0, -1) + J2) * 0.5

            n_i = self.compute_moments('density')
            T_e = self.physical_system.params.fluid_electron_temperature

            # Using a 4th order stencil:
            dn_q1 = (-     af.shift(n_i, 0, 0, -2) + 8 * af.shift(n_i, 0, 0, -1) 
                     - 8 * af.shift(n_i, 0, 0,  1) +     af.shift(n_i, 0, 0,  2)
                    ) / (12 * self.dq1)

            dn_q2 = (-     af.shift(n_i, 0, 0, 0, -2) + 8 * af.shift(n_i, 0, 0, 0, -1) 
                     - 8 * af.shift(n_i, 0, 0, 0,  1) +     af.shift(n_i, 0, 0, 0,  2)
                    ) / (12 * self.dq2)

            # E = -(v X B) + (J X B) / (en) - T ∇n / (en)
            E1 = -v_cross_B_1 + J_cross_B_1 \
                              / (multiply(self.compute_moments('density', f = 0.5 * (self.f + af.shift(self.f, 0, 0, 1))),
                                          self.physical_system.params.charge
                                         )
                                ) \
                              - 0.5 * T_e * (dn_q1 + af.shift(dn_q1, 0, 0, 1)) / multiply(self.physical_system.params.charge, n_i) # (i, j + 1/2)

            E2 = -v_cross_B_2 + J_cross_B_2 \
                              / (multiply(self.compute_moments('density', f = 0.5 * (self.f + af.shift(self.f, 0, 0, 0, 1))),
                                          self.physical_system.params.charge
                                         )
                                ) \
                              - 0.5 * T_e * (dn_q2 + af.shift(dn_q2, 0, 0, 0, 1)) / multiply(self.physical_system.params.charge, n_i) # (i + 1/2, j)

            E3 = -v_cross_B_3 + J_cross_B_3 \
                              / (multiply(self.compute_moments('density', f = self.f),
                                          self.physical_system.params.charge
                                         )
                                ) # (i + 1/2, j + 1/2)
            
            self.fields_solver.yee_grid_EM_fields[0] = E1
            self.fields_solver.yee_grid_EM_fields[1] = E2
            self.fields_solver.yee_grid_EM_fields[2] = E3

            af.eval(self.fields_solver.yee_grid_EM_fields)

        else:
            
            J1 = multiply(self.physical_system.params.charge,
                          self.compute_moments('mom_v1_bulk', f = 0.5 * (self.f + af.shift(self.f, 0, 0, 1)))
                         ) # (i, j + 1/2)

            J2 = multiply(self.physical_system.params.charge,
                          self.compute_moments('mom_v2_bulk', f = 0.5 * (self.f + af.shift(self.f, 0, 0, 0, 1)))
                         ) # (i + 1/2, j)

            J3 = multiply(self.physical_system.params.charge, 
                          self.compute_moments('mom_v3_bulk', f = f)
                         ) # (i + 1/2, j + 1/2)

        self.fields_solver.evolve_electrodynamic_fields(J1, J2, J3, self.dt)

    f_interp_p_3d(self, dt)
    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fieldstep += toc - tic
    
    return
