import arrayfire as af

from bolt.lib.nonlinear_solver.FVM_solver.fluxes \
    import upwind_flux

# Equation to solve:
# df/dt + d(C_q1 * f)/dq1 + d(C_q2 * f)/dq2 = C[f]
# Grid convention considered:

#                  (i, j+1/2)
#              X-------o-------X
#              |               |
#              |               |
#   (i-1/2, j) o       o       o (i+1/2, j)
#              |     (i, j)    |
#              |               |
#              X-------o-------X
#                  (i, j-1/2)

# Using the finite volume method:
# f_{i, j}/dt  = ((- (C_q1 * f)_{i + 1/2, j} + (C_q1 * f)_{i - 1/2, j})/dq1
#                 (- (C_q2 * f)_{i, j + 1/2} + (C_q2 * f)_{i, j - 1/2})/dq2
#                  +  C[f_{i, j}]
#                )

def df_dt(self):

    left_flux, right_flux, bot_flux, top_flux = upwind_flux(self)

    df_dt = - (right_flux - left_flux)/self.dq1
            - (top_flux   - bot_flux )/self.dq2
            + self._source(self.f, self.q1_center, self.q2_center, 
                           self.p1, self.p2, self.p3,
                           self.compute_moments, self.physical_system.params
                          ) 

    af.eval(df_dt)
    return(df_dt)
 