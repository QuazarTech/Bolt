#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np

from bolt.lib.utils.broadcasted_primitive_operations import add, multiply

def f_interp_2d(self, dt):
    """
    Performs 2D interpolation in the q-space to solve for the equation:
    
    df/dt + A_q1 df/dq1 + A_q2 df/dq2 = 0

    This is done by backtracing the characteristic curves
    and interpolating at the origin of the characteristics.

    Parameters
    ----------

    dt : double
         Time-step size to evolve the system
    """
    if(self.performance_test_flag == True):
        tic = af.time()

    A_q1, A_q2 = af.broadcast(self._A_q, self.time_elapsed, 
                              self.q1_center, self.q2_center,
                              self.p1_center, self.p2_center, self.p3_center,
                              self.physical_system.params
                             )

    # Using the add method wrapped with af.broadcast
    q1_center_new = add(self.q1_center, - A_q1 * dt)
    q2_center_new = add(self.q2_center, - A_q2 * dt)

    # Reordering from (dof, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_s, dof)
    # NOTE: To be changed after the implementation of axes specific 
    # interpolation operators gets completed from ArrayFire's end.
    # Ref:https://github.com/arrayfire/arrayfire/issues/1955
    self.f = af.approx2(af.reorder(self.f, 2, 3, 1, 0),
                        af.reorder(q1_center_new, 2, 3, 1, 0),
                        af.reorder(q2_center_new, 2, 3, 1, 0),
                        af.INTERP.BICUBIC_SPLINE, 
                        xp = af.reorder(self.q1_center, 2, 3, 1, 0),
                        yp = af.reorder(self.q2_center, 2, 3, 1, 0)
                       )

    # Reordering from (N_q1, N_q2, N_s, dof) --> (dof, N_s, N_q1, N_q2)
    self.f = af.reorder(self.f, 3, 2, 0, 1)

    af.eval(self.f)
    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_interp2 += toc - tic

    return

def f_interp_p_3d(self, dt):
    """
    Performs 3D interpolation in the p-space to solve for the equation:
    
    df/dt + A_p1 df/dp1 + A_p2 df/dp2 + A_p3 df/dp3 = 0

    Similar to the above function, this is done by backtracing the 
    characteristic curves and interpolating at the origin of the characteristics.
    
    Parameters
    ----------

    dt : double
         Time-step size to evolve the system

    NOTE: This function currently makes use of a Strang split approx1, approx2 with
          reorders to apply along the intended axes. With implementation of approx3
          complete this would be changed to make use of a single call of approx3.
          Ref:https://github.com/arrayfire/arrayfire/issues/1837
    """
    # Following Strang Splitting:
    if(self.performance_test_flag == True):
        tic = af.time()

    A_p1 = af.broadcast(self._A_p, self.time_elapsed,
                        self.q1_center, self.q2_center,
                        self.p1_left, self.p2_left, self.p3_left,
                        self.fields_solver, self.physical_system.params
                       )[0]

    A_p2 = af.broadcast(self._A_p, self.time_elapsed,
                        self.q1_center, self.q2_center,
                        self.p1_bottom, self.p2_bottom, self.p3_bottom,
                        self.fields_solver, self.physical_system.params
                       )[1]

    A_p3 = af.broadcast(self._A_p, self.time_elapsed,
                        self.q1_center, self.q2_center,
                        self.p1_back, self.p2_back, self.p3_back,
                        self.fields_solver, self.physical_system.params
                       )[2]
    
    # # Using the add method wrapped with af.broadcast
    # p1_new = add(self.p1_center, - dt * A_p1)
    # p2_new = add(self.p2_center, - dt * A_p2)

    # # Since the interpolation function are being performed in velocity space,
    # # the arrays used in the computation need to be in p_expanded form.
    # # Hence we will need to convert the same:
    # p1_new = self._convert_to_p_expanded(p1_new)
    # p2_new = self._convert_to_p_expanded(p2_new)

    # # Transforming interpolant to go from [0, N_p - 1]:
    # p1_lower_boundary = add(self.p1_start, 0.5 * self.dp1)
    # p2_lower_boundary = add(self.p2_start, 0.5 * self.dp2)

    # p1_interpolant = multiply(add(p1_new, -p1_lower_boundary), 1 / self.dp1)
    # p2_interpolant = multiply(add(p2_new, -p2_lower_boundary), 1 / self.dp2)

    # if(self.physical_system.params.p_dim == 3):        
        
    #     p3_new = add(self.p3_center, - 0.5 * dt * A_p3)
    #     p3_new = self._convert_to_p_expanded(p3_new)
    #     p3_lower_boundary = add(self.p3_start, 0.5 * self.dp3)
    
    #     # Reordering from (N_p1, N_p2, N_p3, N_s * N_q) --> (N_p3, N_p1, N_p2, N_s * N_q)
    #     p3_interpolant = af.reorder(mulitiply(add(p3_new, -p3_lower_boundary), 1 / self.dp3), 2, 0, 1, 3)

    # # We perform the 3d interpolation by performing individual 1d + 2d interpolations: 
    # self.f = self._convert_to_p_expanded(self.f)
    
    # if(self.physical_system.params.p_dim == 3):
        
    #     # Reordering from (N_p1, N_p2, N_p3, N_s * N_q) --> (N_p3, N_p1, N_p2, N_s * N_q)
    #     self.f = af.approx1(af.reorder(self.f, 2, 0, 1, 3),
    #                         p3_interpolant, 
    #                         af.INTERP.CUBIC_SPLINE
    #                        )

    #     # Reordering back from (N_p1, N_p2, N_p3, N_s * N_q) --> (N_p3, N_p1, N_p2, N_s * N_q)
    #     self.f = af.reorder(self.f, 1, 2, 0, 3)

    # self.f = af.approx2(self.f,
    #                     p1_interpolant,
    #                     p2_interpolant,
    #                     af.INTERP.BICUBIC_SPLINE
    #                    )

    # if(self.physical_system.params.p_dim == 3):
        
    #     # Reordering from (N_p1, N_p2, N_p3, N_s * N_q) --> (N_p3, N_p1, N_p2, N_s * N_q)
    #     self.f = af.approx1(af.reorder(self.f, 2, 0, 1, 3),
    #                         p3_interpolant, 
    #                         af.INTERP.CUBIC_SPLINE
    #                        )

    #     # Reordering back from (N_p3, N_p1, N_p2, N_s * N_q) --> (N_p1, N_p2, N_p3, N_s * N_q)
    #     self.f = af.reorder(self.f, 1, 2, 0, 3)

    # Using the add method wrapped with af.broadcast
    p2_new = add(self.p2_center, - dt * A_p2)
    p3_new = add(self.p3_center, - dt * A_p3)

    # Since the interpolation function are being performed in velocity space,
    # the arrays used in the computation need to be in p_expanded form.
    # Hence we will need to convert the same:
    p2_new = self._convert_to_p_expanded(p2_new)
    p3_new = self._convert_to_p_expanded(p3_new)

    # Transforming interpolant to go from [0, N_p - 1]:
    p2_lower_boundary = add(self.p2_start, 0.5 * self.dp2)
    p3_lower_boundary = add(self.p3_start, 0.5 * self.dp3)

    p2_interpolant = multiply(add(p2_new, -p2_lower_boundary), 1 / self.dp2)
    p3_interpolant = multiply(add(p3_new, -p3_lower_boundary), 1 / self.dp3)

    self.f = self._convert_to_p_expanded(self.f)

    self.f = af.approx2(af.reorder(self.f, 1, 2, 0, 3),
                        af.reorder(p2_interpolant, 1, 2, 0, 3),
                        af.reorder(p3_interpolant, 1, 2, 0, 3),
                        af.INTERP.BICUBIC_SPLINE
                       )

    # Reordering back from (N_p2, N_p3, N_p1, N_s * N_q) --> (N_p1, N_p2, N_p3, N_s * N_q)
    self.f = af.reorder(self.f, 2, 0, 1, 3)

    self.f = self._convert_to_q_expanded(self.f)
    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_interp3 += toc - tic

    return
