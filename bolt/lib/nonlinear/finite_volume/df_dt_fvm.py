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

    if(self.physical_system.params.solver_method_in_q == 'FVM'):

        f_left_plus_eps, f_right_minus_eps = reconstruct(self, f, 2, reconstruction_in_q)
        f_bot_plus_eps, f_top_minus_eps    = reconstruct(self, f, 3, reconstruction_in_q)

        # f_left_minus_eps of i-th cell is f_right_minus_eps of the (i-1)th cell
        f_left_minus_eps = af.shift(f_right_minus_eps, 0, 0, 1)
        # Extending the same to bot:
        f_bot_minus_eps  = af.shift(f_top_minus_eps,   0, 0, 0, 1)

        # af.broadcast used to perform batched operations on arrays of different sizes:
        self._C_q1 = af.broadcast(self._C_q, self.time_elapsed, 
                                  self.q1_left_center, self.q2_left_center,
                                  self.p1_center, self.p2_center, self.p3_center,
                                  self.physical_system.params
                                 )[0]

        self._C_q2 = af.broadcast(self._C_q, self.time_elapsed, 
                                  self.q1_center_bot, self.q2_center_bot,
                                  self.p1_center, self.p2_center, self.p3_center,
                                  self.physical_system.params
                                 )[1]

        f_left = riemann_solver(self, f_left_minus_eps, f_left_plus_eps, self._C_q1)
        f_bot  = riemann_solver(self, f_bot_minus_eps, f_bot_plus_eps, self._C_q2)

        left_flux = multiply(self._C_q1, f_left)
        bot_flux  = multiply(self._C_q2, f_bot)

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
        
        if(    self.physical_system.params.fields_type == 'electrodynamic'
           and self.fields_solver.at_n == False
          ):

            J1 = multiply(self.physical_system.params.charge,
                          self.compute_moments('mom_v1_bulk', f = f_left)
                         ) # (i, j + 1/2)

            J2 = multiply(self.physical_system.params.charge,
                          self.compute_moments('mom_v2_bulk', f = f_bot)
                         ) # (i + 1/2, j)

            J3 = multiply(self.physical_system.params.charge, 
                          self.compute_moments('mom_v3_bulk', f = f)
                         ) # (i + 1/2, j + 1/2)

            self.fields_solver.evolve_electrodynamic_fields(J1, J2, J3, self.dt)


        if(self.physical_system.params.fields_type == 'electrostatic'):
            if(self.physical_system.params.fields_solver == 'fft'):

                rho = multiply(self.physical_system.params.charge,
                               self.compute_moments('density', f = f)
                              )

                self.fields_solver.compute_electrostatic_fields(rho)
        
        self._C_p1 = af.broadcast(self._C_p, self.time_elapsed,
                                  self.q1_center, self.q2_center,
                                  self.p1_left, self.p2_left, self.p3_left,
                                  self.fields_solver, self.physical_system.params
                                 )[0]

        self._C_p2 = af.broadcast(self._C_p, self.time_elapsed,
                                  self.q1_center, self.q2_center,
                                  self.p1_bottom, self.p2_bottom, self.p3_bottom,
                                  self.fields_solver, self.physical_system.params
                                 )[1]

        self._C_p3 = af.broadcast(self._C_p, self.time_elapsed,
                                  self.q1_center, self.q2_center,
                                  self.p1_back, self.p2_back, self.p3_back,
                                  self.fields_solver, self.physical_system.params
                                 )[2]

        self._C_p1 = self._convert_to_p_expanded(self._C_p1)
        self._C_p2 = self._convert_to_p_expanded(self._C_p2)
        self._C_p3 = self._convert_to_p_expanded(self._C_p3)
        f          = self._convert_to_p_expanded(f)
        
        # Variation of p1 is along axis 0:
        f_left_plus_eps, f_right_minus_eps = reconstruct(self, f, 0, reconstruction_in_p)
        # Variation of p2 is along axis 1:
        f_bot_plus_eps, f_top_minus_eps    = reconstruct(self, f, 1, reconstruction_in_p)
        # Variation of p3 is along axis 2:
        f_back_plus_eps, f_front_minus_eps = reconstruct(self, f, 2, reconstruction_in_p)

        # f_left_minus_eps of i-th cell is f_right_minus_eps of the (i-1)th cell
        f_left_minus_eps = af.shift(f_right_minus_eps, 1)
        # Extending the same to bot:
        f_bot_minus_eps  = af.shift(f_top_minus_eps, 0, 1)
        # Extending the same to back:
        f_back_minus_eps = af.shift(f_front_minus_eps, 0, 0, 1)

        # flipping due to strange error seen when working with
        # multiple species on the CPU backend where f != flip(flip(f))
        # Doesn't seem to be a problem on CUDA:
        # Yet to test on OpenCL
        # TODO: Find exact cause for this bug.
        # f != flip(flip(f)) seems to happen after conversion to p_expanded
        # flux_p1 = self._C_p1 * af.flip(af.flip(f))
        # flux_p2 = self._C_p2 * af.flip(af.flip(f))
        # flux_p3 = self._C_p3 * af.flip(af.flip(f))
                
        f_left_p1 = riemann_solver(self, f_left_minus_eps, f_left_plus_eps, self._C_p1)
        f_bot_p2  = riemann_solver(self, f_bot_minus_eps, f_bot_plus_eps, self._C_p2)
        f_back_p3 = riemann_solver(self, f_back_minus_eps, f_back_plus_eps, self._C_p3)
        
        left_flux_p1 = multiply(self._C_p1, f_left_p1)
        bot_flux_p2  = multiply(self._C_p2, f_bot_p2)
        back_flux_p3 = multiply(self._C_p3, f_back_p3)

        right_flux_p1 = af.shift(left_flux_p1, -1)
        top_flux_p2   = af.shift(bot_flux_p2,   0, -1)
        front_flux_p3 = af.shift(back_flux_p3,  0,  0, -1)

        left_flux_p1  = self._convert_to_q_expanded(left_flux_p1)
        right_flux_p1 = self._convert_to_q_expanded(right_flux_p1)

        bot_flux_p2 = self._convert_to_q_expanded(bot_flux_p2)
        top_flux_p2 = self._convert_to_q_expanded(top_flux_p2)

        back_flux_p3  = self._convert_to_q_expanded(back_flux_p3)
        front_flux_p3 = self._convert_to_q_expanded(front_flux_p3)

        df_dt += - (right_flux_p1 - left_flux_p1) / self.dp1 \
                 - (top_flux_p2   - bot_flux_p2 ) / self.dp2 \
                 - (front_flux_p3 - back_flux_p3) / self.dp3

        # By looping over each species:
        # left_flux_p1_all_species = af.constant(0, 512, 2, dtype = af.Dtype.f64)
        # right_flux_p1_all_species = af.constant(0, 512, 2, dtype = af.Dtype.f64)

        # for i in range(self.N_species):

        #     if(i == 1):
        #         # shape of arrays self._C_p1 and f is (512, 1, 1, 2)
        #         # while array[:, :, :, 0] refers to data for species 1
        #         # while array[:, :, :, 1] refers to data for species 2

        #         # Listed below are 3 equivalent methods of calculating flux: 
        #         flux_p1_one   = (af.flip(self._C_p1, 3) * af.flip(f, 3))[:, :, :, 0]
        #         flux_p1_two   = (af.flip(self._C_p1 * f, 3))[:, :, :, 0]
        #         flux_p1_three = (self._C_p1 * f)[:, :, :, 1]
        #         flux_p1_four  = (af.flip(af.flip(self._C_p1, 3), 3) * af.flip(af.flip(f, 3), 3))[:, :, :, 1]

        #         print(af.sum(flux_p1_one == flux_p1_four))  
        #         print(af.sum(flux_p1_one == flux_p1_two))   # True only on first call;false otherwise
        #         print(af.sum(flux_p1_two == flux_p1_three)) # Always true
        #         print(af.sum(flux_p1_one == flux_p1_three)) # True only on first call;false otherwise
        #         print()

        #         # The above shows all to be true on the first call, after which
        #         # it shows only flux_p1_two and flux_p1_three to be matching

        #         # This is what gives the correct answer:
        #         flux_p1 = flux_p1_four
        #         # Using the others produces that odd curve which becomes negative

        #         # self._C_p1 = af.flip(self._C_p1, 3)
        #         # f          = af.flip(f, 3)

        #     else:
        #         flux_p1 = (self._C_p1 * f)[:, :, :, 0]

        #         # self._C_p1 = af.flip(self._C_p1, 3)
        #         # f          = af.flip(f, 3)

        #     # flux_p1 = multiply(self._C_p1, f)[:, :, :, 0]
            
        #     # flux_p2 = multiply(self._C_p2, f)[i * self.N_p1:(i+1) * self.N_p1]
        #     # flux_p3 = multiply(self._C_p3, f)[i * self.N_p1:(i+1) * self.N_p1]

        #     # Variation of p1 is along axis 0:
        #     left_plus_eps_flux_p1   = flux_p1
        #     right_minus_eps_flux_p1 = flux_p1

        #     # left_plus_eps_flux_p1, right_minus_eps_flux_p1 = \
        #     #     reconstruct(self, flux_p1, 0, reconstruction_in_p)
            
        #     # Variation of p2 is along axis 1:
        #     # bot_plus_eps_flux_p2, top_minus_eps_flux_p2 = \
        #     #     reconstruct(self, flux_p2, 1, reconstruction_in_p)

        #     # Variation of p3 is along axis 2:
        #     # back_plus_eps_flux_p3, front_minus_eps_flux_p3 = \
        #     #     reconstruct(self, flux_p3, 2, reconstruction_in_p)

        #     left_minus_eps_flux_p1 = af.shift(right_minus_eps_flux_p1, 1)
        #     # bot_minus_eps_flux_p2  = af.shift(top_minus_eps_flux_p2,   0, 1)
        #     # back_minus_eps_flux_p3 = af.shift(front_minus_eps_flux_p3, 0, 0, 1)

        #     # Averaging just for a visual check:
        #     left_flux_p1 = 0.5 * (left_minus_eps_flux_p1 + left_plus_eps_flux_p1)
        #     # left_flux_p1 = riemann_solver(self, left_minus_eps_flux_p1, left_plus_eps_flux_p1,
        #     #                               f_left_minus_eps, f_left_plus_eps, riemann_in_p, 'p1'
        #     #                              )

        #     # bot_flux_p2  = riemann_solver(self, bot_minus_eps_flux_p2, bot_plus_eps_flux_p2,
        #     #                               f_bot_minus_eps, f_bot_plus_eps, riemann_in_p, 'p2'
        #     #                              )

        #     # back_flux_p3 = riemann_solver(self, back_minus_eps_flux_p3, back_plus_eps_flux_p3,
        #     #                               f_back_plus_eps, f_back_minus_eps, riemann_in_p, 'p3'
        #     #                              )

        #     right_flux_p1 = af.shift(left_flux_p1, -1)
        #     # top_flux_p2   = af.shift(bot_flux_p2,   0, -1)
        #     # front_flux_p3 = af.shift(back_flux_p3,  0,  0, -1)

        #     left_flux_p1_all_species[:, i]  = af.moddims(left_flux_p1, 512)
        #     right_flux_p1_all_species[:, i] = af.moddims(right_flux_p1, 512)

        #     # if(i == 1):
        #     #     self._C_p1 = af.flip(self._C_p1, 3)
        #     #     f          = af.flip(f, 3)

        #     # left_flux_p1  = self._convert_to_q_expanded(left_flux_p1)
        #     # right_flux_p1 = self._convert_to_q_expanded(right_flux_p1)

        #     # bot_flux_p2 = self._convert_to_q_expanded(bot_flux_p2)
        #     # top_flux_p2 = self._convert_to_q_expanded(top_flux_p2)

        #     # back_flux_p3  = self._convert_to_q_expanded(back_flux_p3)
        #     # front_flux_p3 = self._convert_to_q_expanded(front_flux_p3)

        # df_dt += - (right_flux_p1_all_species - left_flux_p1_all_species)/self.dp1 \

    af.eval(df_dt)
    return(df_dt)
