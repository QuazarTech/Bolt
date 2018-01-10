import arrayfire as af

# Importing Riemann solver used in calculating fluxes:
from .riemann_solver import riemann_solver, upwind_flux
from .reconstruct import reconstruct

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

@af.broadcast
def multiply(a, b):
    return(a * b)


def df_dt_fvm(f, self):
    
    # Giving shorter name references:
    reconstruction_in_q = self.physical_system.params.reconstruction_method_in_q
    reconstruction_in_p = self.physical_system.params.reconstruction_method_in_p
    
    # Initializing df_dt
    df_dt = 0

    self._C_q1, self._C_q2 = \
        af.broadcast(self._C_q, self.f, self.time_elapsed, 
                     self.q1_center, self.q2_center,
                     self.p1_center, self.p2_center, self.p3_center,
                     self.physical_system.params
                    )

    if(self.physical_system.params.solver_method_in_q == 'FVM'):
        
        # Variation of q1 is along axis 1
        left_plus_eps_flux, right_minus_eps_flux = \
            reconstruct(self, multiply(self._C_q1, f), 2, reconstruction_in_q)
        
        # Variation of q2 is along axis 2
        bot_plus_eps_flux, top_minus_eps_flux = \
            reconstruct(self, multiply(self._C_q2, f), 3, reconstruction_in_q)

        if(self.physical_system.params.riemann_solver_in_q == 'lax-friedrichs'):
            f_left_plus_eps, f_right_minus_eps = reconstruct(self, f, 2, reconstruction_in_q)
            f_bot_plus_eps, f_top_minus_eps    = reconstruct(self, f, 3, reconstruction_in_q)
    
            # f_left_minus_eps of i-th cell is f_right_minus_eps of the (i-1)th cell
            f_left_minus_eps = af.shift(f_right_minus_eps, 0, 0, 1)
            # Extending the same to bot:
            f_bot_minus_eps  = af.shift(f_top_minus_eps,   0, 0, 0, 1)

        else:
            f_left_plus_eps, f_left_minus_eps = 0, 0
            f_bot_plus_eps,  f_bot_minus_eps  = 0, 0 

        # Applying the shifts to the fluxes:
        left_minus_eps_flux = af.shift(right_minus_eps_flux, 0, 0, 1)
        bot_minus_eps_flux  = af.shift(top_minus_eps_flux,   0, 0, 0, 1)

        left_flux  = riemann_solver(self, left_minus_eps_flux, left_plus_eps_flux,
                                    f_left_minus_eps, f_left_plus_eps, 'q1'
                                   )

        bot_flux   = riemann_solver(self, bot_minus_eps_flux, bot_plus_eps_flux,
                                    f_bot_minus_eps, f_bot_plus_eps, 'q2'
                                   )

        right_flux = af.shift(left_flux, 0, 0, -1)
        top_flux   = af.shift(bot_flux,  0, 0,  0, -1)
        
        df_dt += - (right_flux - left_flux)/self.dq1 \
                 - (top_flux   - bot_flux )/self.dq2 \

        if(self.physical_system.params.source_enabled == True):

            df_dt += self._source(f, self.time_elapsed, 
                                  self.q1_center, self.q2_center,
                                  self.p1_center, self.p2_center, self.p3_center, 
                                  self.compute_moments, 
                                  self.physical_system.params, False
                                 ) 

    if(    self.physical_system.params.solver_method_in_p == 'FVM' 
       and self.physical_system.params.EM_fields_enabled == True
      ):

        if(self.physical_system.params.fields_solver == 'fft'):
            rho = multiply(self.physical_system.params.charge,
                           self.compute_moments('density', f=f)
                          )
            self.fields_solver.compute_electrostatic_fields(rho)

        (C_p1, C_p2, C_p3) = af.broadcast(self._C_p, self.f, self.time_elapsed,
                                          self.q1_center, self.q2_center,
                                          self.p1_center, self.p2_center, self.p3_center,
                                          self.fields_solver, self.physical_system.params
                                         )

        flux_p1 = self._convert_to_p_expanded(multiply(C_p1, f))
        flux_p2 = self._convert_to_p_expanded(multiply(C_p2, f))
        flux_p3 = self._convert_to_p_expanded(multiply(C_p3, f))

        N_g_p = self.N_ghost_p

        # Setting flux values in the ghost zones to zero:
        if(N_g_p != 0):
            
            flux_p1[:N_g_p]        = 0
            flux_p1[:, :N_g_p]     = 0
            flux_p1[:, :, :N_g_p]  = 0
            flux_p1[-N_g_p:]       = 0 * flux_p1[-N_g_p:]
            flux_p1[:, -N_g_p:]    = 0 * flux_p1[:, -N_g_p:]
            flux_p1[:, :, -N_g_p:] = 0 * flux_p1[:, :, -N_g_p:]

            flux_p2[:N_g_p]        = 0
            flux_p2[:, :N_g_p]     = 0
            flux_p2[:, :, :N_g_p]  = 0
            flux_p2[-N_g_p:]       = 0 * flux_p2[-N_g_p:]
            flux_p2[:, -N_g_p:]    = 0 * flux_p2[:, -N_g_p:]
            flux_p2[:, :, -N_g_p:] = 0 * flux_p2[:, :, -N_g_p:]

            flux_p3[:N_g_p]        = 0
            flux_p3[:, :N_g_p]     = 0
            flux_p3[:, :, :N_g_p]  = 0
            flux_p3[-N_g_p:]       = 0 * flux_p3[-N_g_p:]
            flux_p3[:, -N_g_p:]    = 0 * flux_p3[:, -N_g_p:]
            flux_p3[:, :, -N_g_p:] = 0 * flux_p3[:, :, -N_g_p:]

        # Variation of p1 is along axis 0:
        left_plus_eps_flux_p1, right_minus_eps_flux_p1 = \
            reconstruct(self, flux_p1, 0, reconstruction_in_p)
        # Variation of p2 is along axis 1:
        bot_plus_eps_flux_p2, top_minus_eps_flux_p2 = \
            reconstruct(self, flux_p2, 1, reconstruction_in_p)
        # Variation of p3 is along axis 2:
        back_plus_eps_flux_p3, front_minus_eps_flux_p3 = \
            reconstruct(self, flux_p3, 2, reconstruction_in_p)

        left_minus_eps_flux_p1 = af.shift(right_minus_eps_flux_p1, 1)
        bot_minus_eps_flux_p2  = af.shift(top_minus_eps_flux_p2,   0, 1)
        back_minus_eps_flux_p3 = af.shift(front_minus_eps_flux_p3, 0, 0, 1)


        if(self.physical_system.params.riemann_solver_in_p == 'averaged-flux'):
            
            left_flux_p1  = 0.5 * (left_minus_eps_flux_p1 + left_plus_eps_flux_p1)
            bot_flux_p2   = 0.5 * (bot_minus_eps_flux_p2  + bot_plus_eps_flux_p2)
            back_flux_p3  = 0.5 * (back_minus_eps_flux_p3 + back_plus_eps_flux_p3)

        elif(self.physical_system.params.riemann_solver_in_p == 'upwind-flux'):

            left_flux_p1 = upwind_flux(left_minus_eps_flux_p1, 
                                       left_plus_eps_flux_p1,
                                       self._convert_to_p_expanded(C_p1) 
                                      )

            bot_flux_p2 = upwind_flux(bot_minus_eps_flux_p2, 
                                      bot_plus_eps_flux_p2,
                                      self._convert_to_p_expanded(C_p2) 
                                     )

            back_flux_p3 = upwind_flux(back_minus_eps_flux_p3, 
                                       back_plus_eps_flux_p3,
                                       self._convert_to_p_expanded(C_p3) 
                                      )

        right_flux_p1 = af.shift(left_flux_p1, -1)
        top_flux_p2   = af.shift(bot_flux_p2,   0, -1)
        front_flux_p3 = af.shift(back_flux_p3,  0,  0, -1)

        left_flux_p1  = self._convert_to_q_expanded(left_flux_p1)
        right_flux_p1 = self._convert_to_q_expanded(right_flux_p1)

        bot_flux_p2   = self._convert_to_q_expanded(bot_flux_p2)
        top_flux_p2   = self._convert_to_q_expanded(top_flux_p2)

        back_flux_p3  = self._convert_to_q_expanded(back_flux_p3)
        front_flux_p3 = self._convert_to_q_expanded(front_flux_p3)

        df_dt += - (right_flux_p1 - left_flux_p1)/self.dp1 \
                 - (top_flux_p2   - bot_flux_p2 )/self.dp2 \
                 - (front_flux_p3 - back_flux_p3)/self.dp3
    
    af.eval(df_dt)
    return(df_dt)
