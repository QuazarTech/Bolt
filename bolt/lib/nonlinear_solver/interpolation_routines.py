#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af


def f_interp_2d(self, dt):
    # Obtaining the left-bottom corner coordinates
    # (lowest values of the canonical coordinates in the local zone)
    # Additionally, we also obtain the size of the local zone
    ((i_q1_lowest, i_q2_lowest), (N_q1_local,
                                  N_q2_local)) = self._da.getCorners()

    addition = lambda a, b:a + b

    q1_center_new = af.broadcast(addition, self.q1_center, - self._A_q1 * dt)
    q2_center_new = af.broadcast(addition, self.q2_center, - self._A_q2 * dt)

    # Obtaining the center coordinates:
    (i_q1_center, i_q2_center) = (i_q1_lowest + 0.5, i_q2_lowest + 0.5)

    # Obtaining the left, and bottom boundaries for the local zones:
    q1_boundary_lower = self.q1_start + i_q1_center * self.dq1
    q2_boundary_lower = self.q2_start + i_q2_center * self.dq2

    # Adding N_ghost to account for the offset due to ghost zones:
    q1_interpolant = (q1_center_new - q1_boundary_lower) / self.dq1 +\
                     self.N_ghost
    q2_interpolant = (q2_center_new - q2_boundary_lower) / self.dq2 +\
                     self.N_ghost

    self.f = af.approx2(self.f, q1_interpolant, q2_interpolant,
                        af.INTERP.BICUBIC_SPLINE)

    af.eval(self.f)
    return


def f_interp_p_3d(self, dt):
    """
    Since the interpolation function are being performed in velocity space,
    the arrays used in the computation need to be in velocitiesExpanded form.
    Hence we will need to convert the same:
    """
    # Following Lie Splitting:
    (A_p1, A_p2, A_p3) = af.broadcast(self._A_p, self.q1_center, self.q2_center,
                                      self.p1, self.p2, self.p3,
                                      self.E1, self.E2, self.E3,
                                      self.B1, self.B2, self.B3,
                                      self.physical_system.params
                                     )
    
    addition = lambda a,b:a + b
    
    p1_new = af.broadcast(addition, self.p1, - 0.5 * dt * A_p1)
    p2_new = af.broadcast(addition, self.p2, - dt * A_p2)
    p3_new = af.broadcast(addition, self.p3, - dt * A_p3)

    p1_new = self._convert_to_pExpanded(p1_new)
    p2_new = self._convert_to_pExpanded(p2_new)
    p3_new = self._convert_to_pExpanded(p3_new)

    # Transforming interpolant to go from [0, N_p - 1]:
    p1_lower_boundary = self.p1_start + 0.5 * self.dp1
    p2_lower_boundary = self.p2_start + 0.5 * self.dp2
    p3_lower_boundary = self.p3_start + 0.5 * self.dp3

    p1_interpolant = (p1_new - p1_lower_boundary) / self.dp1
    p2_interpolant = (p2_new - p2_lower_boundary) / self.dp2
    p3_interpolant = (p3_new - p3_lower_boundary) / self.dp3

    # We perform the 3d interpolation by performing
    # individual 1d + 2d interpolations. Reordering to bring the
    # variation in values along axis 0 and axis 1

    self.f = self._convert_to_pExpanded(self.f)

    self.f = af.approx1(af.reorder(self.f),
                        af.reorder(p1_interpolant),
                        af.INTERP.CUBIC_SPLINE)

    self.f = af.approx2(af.reorder(self.f, 2, 3, 1, 0),
                        af.reorder(p2_interpolant, 2, 3, 0, 1),
                        af.reorder(p3_interpolant, 2, 3, 0, 1),
                        af.INTERP.BICUBIC_SPLINE)

    self.f = af.approx1(af.reorder(self.f, 3, 2, 0, 1),
                        af.reorder(p1_interpolant),
                        af.INTERP.CUBIC_SPLINE)

    self.f = af.reorder(self.f)
    self.f = self._convert_to_qExpanded(self.f)

    af.eval(self.f)
    return
