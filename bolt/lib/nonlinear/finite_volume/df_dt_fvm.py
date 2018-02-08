import arrayfire as af

# Importing Riemann solver used in calculating fluxes:
from .riemann import riemann_solver
from .reconstruct import reconstruct
from bolt.lib.utils.broadcasted_primitive_operations import multiply

"""
Equation to solve:
When solving only for q-space:
df/dt + d(C_q1 * f)/dq1 + d(C_q2 * f)/dq2 = C[f]
Grid convention considered:

                 (i+1/2, j+1)
             X-------o-------X
             |               |
             |               |
  (i, j+1/2) o       o       o (i+1, j+1/2)
             | (i+1/2, j+1/2)|
             |               |
             X-------o-------X
                 (i+1/2, j)

Using the finite volume method in q-space:
d(f_{i+1/2, j+1/2})/dt  = ((- (C_q1 * f)_{i + 1, j + 1/2} + (C_q1 * f)_{i, j + 1/2})/dq1
                           (- (C_q2 * f)_{i + 1/2, j + 1} + (C_q2 * f)_{i + 1/2, j})/dq2
                           +  C[f_{i+1/2, j+1/2}]
                          )
The same concept is extended to p-space as well.                          
"""

def df_dt_fvm(f, self):
    """
    Returns the expression for df/dt which is then 
    evolved by a timestepper.

    Parameters
    ----------

    f : af.Array
        Array of the distribution function at which df_dt is to 
        be evaluated.
    """ 
    
    # Giving shorter name references:
    reconstruction_in_q = self.physical_system.params.reconstruction_method_in_q
    reconstruction_in_p = self.physical_system.params.reconstruction_method_in_p
    
    riemann_in_q = self.physical_system.params.riemann_solver_in_q
    riemann_in_p = self.physical_system.params.riemann_solver_in_p

    # Initializing df_dt
    df_dt = 0

    # af.broadcast used to perform batched operations on arrays of different sizes:
    self._C_q1, self._C_q2 = \
        af.broadcast(self._C_q, f, self.time_elapsed, 
                     self.q1_center, self.q2_center,
                     self.p1_center, self.p2_center, self.p3_center,
                     self.physical_system.params
                    )

    if(self.physical_system.params.solver_method_in_q == 'FVM'):
        
        # Variation of q1 is along axis 2
        left_plus_eps_flux, right_minus_eps_flux = \
            reconstruct(self, multiply(self._C_q1, f), 2, reconstruction_in_q)
        
        # Variation of q2 is along axis 3
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
                                    f_left_minus_eps, f_left_plus_eps, riemann_in_q, 'q1'
                                   )

        bot_flux   = riemann_solver(self, bot_minus_eps_flux, bot_plus_eps_flux,
                                    f_bot_minus_eps, f_bot_plus_eps, riemann_in_q, 'q2'
                                   )

        right_flux = af.shift(left_flux, 0, 0, -1)
        top_flux   = af.shift(bot_flux,  0, 0,  0, -1)
        
        df_dt += - (right_flux - left_flux)/self.dq1 \
                 - (top_flux   - bot_flux )/self.dq2 \

        if(    self.physical_system.params.source_enabled == True 
           and self.physical_system.params.instantaneous_collisions != True
          ):
            df_dt += self._source(f, self.time_elapsed, 
                                  self.q1_center, self.q2_center,
                                  self.p1_center, self.p2_center, self.p3_center, 
                                  self.compute_moments, 
                                  self.physical_system.params, False
                                 ) 

    if(    self.physical_system.params.solver_method_in_p == 'FVM' 
       and self.physical_system.params.fields_enabled == True
      ):

        if(self.physical_system.params.fields_type == 'electrostatic'):
            if(self.physical_system.params.fields_solver == 'fft'):
                rho = multiply(self.physical_system.params.charge,
                               self.compute_moments('density', f=f)
                              )
                self.fields_solver.compute_electrostatic_fields(rho)

        (self._C_p1, self._C_p2, self._C_p3) = \
            af.broadcast(self._C_p, f, self.time_elapsed,
                         self.q1_center, self.q2_center,
                         self.p1_center, self.p2_center, self.p3_center,
                         self.fields_solver, self.physical_system.params
                        )

        self._C_p1 = self._convert_to_p_expanded(self._C_p1)
        self._C_p2 = self._convert_to_p_expanded(self._C_p2)
        self._C_p3 = self._convert_to_p_expanded(self._C_p3)
        f          = self._convert_to_p_expanded(f)

        if(self.physical_system.params.riemann_solver_in_p == 'lax-friedrichs'):
            
            f_left_plus_eps, f_right_minus_eps = reconstruct(self, f, 0, reconstruction_in_p)
            f_bot_plus_eps, f_top_minus_eps    = reconstruct(self, f, 1, reconstruction_in_p)
            f_back_plus_eps, f_front_minus_eps = reconstruct(self, f, 2, reconstruction_in_p)
    
            # f_left_minus_eps of i-th cell is f_right_minus_eps of the (i-1)th cell
            f_left_minus_eps = af.shift(f_right_minus_eps, 1)
            # Extending the same to bot:
            f_bot_minus_eps  = af.shift(f_top_minus_eps, 0, 1)
            # Extending the same to back:
            f_back_minus_eps = af.shift(f_front_minus_eps, 0, 0, 1)

        else:
            f_left_plus_eps, f_left_minus_eps  = 0, 0
            f_bot_plus_eps,  f_bot_minus_eps   = 0, 0 
            f_back_plus_eps, f_back_minus_eps = 0, 0 

        flux_p1 = multiply(self._C_p1, f)
        flux_p2 = multiply(self._C_p2, f)
        flux_p3 = multiply(self._C_p3, f)

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

        left_flux_p1 = riemann_solver(self, left_minus_eps_flux_p1, left_plus_eps_flux_p1,
                                      f_left_minus_eps, f_left_plus_eps, riemann_in_p, 'p1'
                                     )

        bot_flux_p2  = riemann_solver(self, bot_minus_eps_flux_p2, bot_plus_eps_flux_p2,
                                      f_bot_minus_eps, f_bot_plus_eps, riemann_in_p, 'p2'
                                     )

        back_flux_p3 = riemann_solver(self, back_minus_eps_flux_p3, back_plus_eps_flux_p3,
                                      f_back_plus_eps, f_back_minus_eps, riemann_in_p, 'p3'
                                     )

        right_flux_p1 = af.shift(left_flux_p1, -1)
        top_flux_p2   = af.shift(bot_flux_p2,   0, -1)
        front_flux_p3 = af.shift(back_flux_p3,  0,  0, -1)

        left_flux_p1  = self._convert_to_q_expanded(left_flux_p1)
        right_flux_p1 = self._convert_to_q_expanded(right_flux_p1)

        bot_flux_p2 = self._convert_to_q_expanded(bot_flux_p2)
        top_flux_p2 = self._convert_to_q_expanded(top_flux_p2)

        back_flux_p3  = self._convert_to_q_expanded(back_flux_p3)
        front_flux_p3 = self._convert_to_q_expanded(front_flux_p3)

        df_dt += - (right_flux_p1 - left_flux_p1)/self.dp1 \
                 - (top_flux_p2   - bot_flux_p2 )/self.dp2 \
                 - (front_flux_p3 - back_flux_p3)/self.dp3

    af.eval(df_dt)
    return(df_dt)
