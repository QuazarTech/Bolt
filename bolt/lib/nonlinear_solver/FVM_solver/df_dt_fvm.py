import arrayfire as af

from bolt.lib.nonlinear_solver.EM_fields_solver.electrostatic import fft_poisson
from bolt.lib.nonlinear_solver.EM_fields_solver.electrostatic import compute_electrostatic_fields
# Importing Riemann solver used in calculating fluxes:
from .riemann_solver import riemann_solver, upwind_flux
from .reconstruct import reconstruct
import params

# Equation to solve:
# df/dt + d(C_q1 * f)/dq1 + d(C_q2 * f)/dq2 = C[f]
# Grid convention considered:

#                  (i+1/2, j+1)
#              X-------o-------X
#              |               |
#              |               |
#   (i, j+1/2) o       o       o (i+1, j+1/2)
#              | (i+1/2, j+1/2)|
#              |               |
#              X-------o-------X
#                  (i+1/2, j)

# Using the finite volume method:
# d(f_{i+1/2, j+1/2})/dt  = ((- (C_q1 * f)_{i + 1, j + 1/2} + (C_q1 * f)_{i, j + 1/2})/dq1
                          #  (- (C_q2 * f)_{i + 1/2, j + 1} + (C_q2 * f)_{i + 1/2, j})/dq2
                          #  +  C[f_{i+1/2, j+1/2}]
                          # )

def df_dt_fvm(f, self, at_n = True):
    
    multiply = lambda a, b: a * b

    # Giving shorter name references:
    method_in_q = self.physical_system.params.reconstruction_method_in_q
    method_in_p = self.physical_system.params.reconstruction_method_in_p

    # Variation of q1 is along axis 1
    left_plus_eps_flux, right_minus_eps_flux = \
        reconstruct(self, af.broadcast(multiply, self._C_q1, f), 1, method_in_q)
    
    # Variation of q2 is along axis 2
    bot_plus_eps_flux, top_minus_eps_flux = \
        reconstruct(self, af.broadcast(multiply, self._C_q2, f), 2, method_in_q)

    f_left_plus_eps, f_right_minus_eps = reconstruct(self, f, 1, method_in_q)
    f_bot_plus_eps, f_top_minus_eps    = reconstruct(self, f, 2, method_in_q)

    # f_left_minus_eps of i-th cell is f_right_minus_eps of the (i-1)th cell
    f_left_minus_eps = af.shift(f_right_minus_eps, 0,  1)
    # Extending the same to bot:
    f_bot_minus_eps  = af.shift(f_top_minus_eps,   0,  0, 1)

    # Applying the shifts to the fluxes:
    left_minus_eps_flux = af.shift(right_minus_eps_flux, 0,  1)
    bot_minus_eps_flux  = af.shift(top_minus_eps_flux,   0,  0,  1)

    left_flux  = riemann_solver(self, left_minus_eps_flux, left_plus_eps_flux,
                                f_left_minus_eps, f_left_plus_eps, 'q1'
                               )

    bot_flux   = riemann_solver(self, bot_minus_eps_flux, bot_plus_eps_flux,
                                f_bot_minus_eps, f_bot_plus_eps, 'q2'
                               )

    right_flux = af.shift(left_flux, 0, -1)
    top_flux   = af.shift(bot_flux,  0,  0, -1)
    
    df_dt = - (right_flux - left_flux)/self.dq1 \
            - (top_flux   - bot_flux )/self.dq2 \
            + self._source(f, self.q1_center, self.q2_center,
                           self.p1, self.p2, self.p3, 
                           self.compute_moments, 
                           self.physical_system.params, False
                          ) 

    if(    self.physical_system.params.solver_method_in_p == 'FVM' 
       and self.physical_system.params.charge_electron != 0
      ):
        if(self.physical_system.params.fields_type == 'electrostatic'):

            if (params.time_step%params.electrostatic_solver_every_nth_step==0):

                if(self.physical_system.params.fields_solver == 'fft'):

                    fft_poisson(self, f)
            
                elif(self.physical_system.params.fields_solver == 'SNES'):
                    compute_electrostatic_fields(self)
                    pass

            E1 = self.cell_centered_EM_fields[0]
            E2 = self.cell_centered_EM_fields[1]
            E3 = self.cell_centered_EM_fields[2]

            B1 = self.cell_centered_EM_fields[3]
            B2 = self.cell_centered_EM_fields[4]
            B3 = self.cell_centered_EM_fields[5]

        # This is taken care of by the timestepper that is utilized
        # when FDTD is to be used with FVM in p-space
        elif(self.physical_system.params.fields_solver == 'fdtd'):
            pass

        else:
            raise NotImplementedError('The method specified is \
                                       invalid/not-implemented'
                                     )

#        if(    self.physical_system.params.fields_solver == 'fdtd'
#           and at_n == True
#          ):
#
#            E1 = self.cell_centered_EM_fields_at_n[0]
#            E2 = self.cell_centered_EM_fields_at_n[1]
#            E3 = self.cell_centered_EM_fields_at_n[2]
#
#            B1 = self.cell_centered_EM_fields_at_n[3]
#            B2 = self.cell_centered_EM_fields_at_n[4]
#            B3 = self.cell_centered_EM_fields_at_n[5]
#
#        elif(    self.physical_system.params.fields_solver == 'fdtd'
#             and at_n != False
#            ):
#
#            E1 = self.cell_centered_EM_fields_at_n_plus_half[0]
#            E2 = self.cell_centered_EM_fields_at_n_plus_half[1]
#            E3 = self.cell_centered_EM_fields_at_n_plus_half[2]
#
#            B1 = self.cell_centered_EM_fields_at_n_plus_half[3]
#            B2 = self.cell_centered_EM_fields_at_n_plus_half[4]
#            B3 = self.cell_centered_EM_fields_at_n_plus_half[5]
#
#        else:
#
#            E1 = self.cell_centered_EM_fields[0]
#            E2 = self.cell_centered_EM_fields[1]
#            E3 = self.cell_centered_EM_fields[2]
#
#            B1 = self.cell_centered_EM_fields[3]
#            B2 = self.cell_centered_EM_fields[4]
#            B3 = self.cell_centered_EM_fields[5]

        (A_p1, A_p2, A_p3) = af.broadcast(self._A_p, self.q1_center, self.q2_center,
                                          self.p1, self.p2, self.p3,
                                          E1, E2, E3, B1, B2, B3,
                                          self.physical_system.params
                                         )

        # Variation of p1 is along axis 0:
        left_plus_eps_flux_p1, right_minus_eps_flux_p1 = \
            reconstruct(self, 
                        self._convert_to_p_expanded(af.broadcast(multiply, A_p1, f)),
                        0, method_in_p
                       )
        # Variation of p2 is along axis 1:
        bot_plus_eps_flux_p2, top_minus_eps_flux_p2 = \
            reconstruct(self,
                        self._convert_to_p_expanded(af.broadcast(multiply, A_p2, f)),
                        1, method_in_p
                       )
#        # Variation of p3 is along axis 2:
#        back_plus_eps_flux_p3, front_minus_eps_flux_p3 = \
#            reconstruct(self, self._convert_to_p_expanded(af.broadcast(multiply, A_p3, f)), 2, method_in_p)

        left_minus_eps_flux_p1 = af.shift(right_minus_eps_flux_p1, 1)
        bot_minus_eps_flux_p2  = af.shift(top_minus_eps_flux_p2,   0, 1)
#        back_minus_eps_flux_p3 = af.shift(front_minus_eps_flux_p3, 0, 0, 1)

        # Obtaining the fluxes by face-averaging:
#        left_flux_p1  = 0.5 * (left_minus_eps_flux_p1 + left_plus_eps_flux_p1)
#        bot_flux_p2   = 0.5 * (bot_minus_eps_flux_p2  + bot_plus_eps_flux_p2)
 #       back_flux_p3  = 0.5 * (back_minus_eps_flux_p3 + back_plus_eps_flux_p3)

        left_flux_p1 = upwind_flux(left_minus_eps_flux_p1, \
                                   left_plus_eps_flux_p1, \
                                   self._convert_to_p_expanded(A_p1)
                                  )

        bot_flux_p2  = upwind_flux(bot_minus_eps_flux_p2, \
                                   bot_plus_eps_flux_p2, \
                                   self._convert_to_p_expanded(A_p2)
                                  )

        right_flux_p1 = af.shift(left_flux_p1, -1)
        top_flux_p2   = af.shift(bot_flux_p2,   0, -1)
#        front_flux_p3 = af.shift(back_flux_p3,  0,  0, -1)

        left_flux_p1  = self._convert_to_q_expanded(left_flux_p1)
        right_flux_p1 = self._convert_to_q_expanded(right_flux_p1)

        bot_flux_p2   = self._convert_to_q_expanded(bot_flux_p2)
        top_flux_p2   = self._convert_to_q_expanded(top_flux_p2)

#        back_flux_p3  = self._convert_to_q_expanded(back_flux_p3)
#        front_flux_p3 = self._convert_to_q_expanded(front_flux_p3)

        df_dt += - (right_flux_p1 - left_flux_p1)/self.dp1 \
                 - (top_flux_p2   - bot_flux_p2 )/self.dp2 \
                 #- (front_flux_p3 - back_flux_p3)/self.dp3
    
    af.eval(df_dt)
    return(df_dt)
 
