import arrayfire as af

# Importing Riemann solver used in calculating fluxes:
from .riemann_solver import riemann_solver

from .reconstruction.minmod import reconstruct_minmod
from .reconstruction.ppm import reconstruct_ppm
from .reconstruction.weno5 import reconstruct_weno5

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
# d(f_{i, j})/dt  = ((- (C_q1 * f)_{i + 1/2, j} + (C_q1 * f)_{i - 1/2, j})/dq1
#                   (- (C_q2 * f)_{i, j + 1/2} + (C_q2 * f)_{i, j - 1/2})/dq2
#                    +  C[f_{i, j}]
#                   )

def df_dt(f, self):
    
    multiply = lambda a, b: a * b

    if(self.performance_test_flag == True):
        tic = af.time()

    if(self.physical_system.params.reconstruction_method == 'piecewise-constant'):

        left_plus_eps_flux   = af.broadcast(multiply, self._C_q1, f)
        right_minus_eps_flux = af.broadcast(multiply, self._C_q1, f)
        bot_plus_eps_flux    = af.broadcast(multiply, self._C_q2, f)
        top_minus_eps_flux   = af.broadcast(multiply, self._C_q2, f)

        f_left_plus_eps   = f
        f_right_minus_eps = f
        f_bot_plus_eps    = f
        f_top_minus_eps   = f


    elif(self.physical_system.params.reconstruction_method == 'minmod'):
        
        left_plus_eps_flux, right_minus_eps_flux = \
            reconstruct_minmod(af.broadcast(multiply, self._C_q1, f), 'q1')
        bot_plus_eps_flux, top_minus_eps_flux = \
            reconstruct_minmod(af.broadcast(multiply, self._C_q2, f), 'q2')

        f_left_plus_eps, f_right_minus_eps = reconstruct_minmod(f, 'q1')
        f_bot_plus_eps, f_top_minus_eps    = reconstruct_minmod(f, 'q2')
        
    elif(self.physical_system.params.reconstruction_method == 'ppm'):
        
        left_plus_eps_flux, right_minus_eps_flux = \
            reconstruct_ppm(af.broadcast(multiply, self._C_q1, f), 'q1')
        bot_plus_eps_flux, top_minus_eps_flux = \
            reconstruct_ppm(af.broadcast(multiply, self._C_q2, f), 'q2')

        f_left_plus_eps, f_right_minus_eps = reconstruct_ppm(f, 'q1')
        f_bot_plus_eps, f_top_minus_eps    = reconstruct_ppm(f, 'q2')

    elif(self.physical_system.params.reconstruction_method == 'weno5'):
        
        left_plus_eps_flux, right_minus_eps_flux = \
            reconstruct_weno5(af.broadcast(multiply, self._C_q1, f), 'q1')
        bot_plus_eps_flux, top_minus_eps_flux = \
            reconstruct_weno5(af.broadcast(multiply, self._C_q2, f), 'q2')

        f_left_plus_eps, f_right_minus_eps = reconstruct_weno5(f, 'q1')
        f_bot_plus_eps, f_top_minus_eps    = reconstruct_weno5(f, 'q2')

    else:
        raise NotImplementedError('Reconstruction method invalid/not-implemented')
    
    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_reconstruct += toc - tic

    # f_left_minus_eps of i-th cell is f_right_minus_eps of the (i-1)th cell
    # f_right_plus_eps of i-th cell is f_left_plus_eps   of the (i+1)th cell
    f_left_minus_eps = af.shift(f_right_minus_eps, 0,  1)
    f_right_plus_eps = af.shift(f_left_plus_eps,   0, -1)

    # Extending the same to bot/top:
    f_bot_minus_eps  = af.shift(f_top_minus_eps, 0, 0,  1)
    f_top_plus_eps   = af.shift(f_bot_plus_eps,  0, 0, -1)

    # Applying the shifts to the fluxes:
    left_minus_eps_flux = af.shift(right_minus_eps_flux, 0,  1)
    right_plus_eps_flux = af.shift(left_plus_eps_flux,   0, -1)
    bot_minus_eps_flux  = af.shift(top_minus_eps_flux, 0, 0,  1)
    top_plus_eps_flux   = af.shift(bot_plus_eps_flux,  0, 0, -1)

    left_flux  = riemann_solver(self, left_minus_eps_flux, left_plus_eps_flux,
                                f_left_minus_eps, f_left_plus_eps
                               )

    right_flux = riemann_solver(self, right_minus_eps_flux, right_plus_eps_flux,
                                f_right_minus_eps, f_right_plus_eps
                               )

    bot_flux   = riemann_solver(self, bot_minus_eps_flux, bot_plus_eps_flux,
                                f_bot_minus_eps, f_bot_plus_eps
                               )

    top_flux   = riemann_solver(self, top_minus_eps_flux, top_plus_eps_flux,
                                f_top_minus_eps, f_top_plus_eps
                               )

    df_dt = - (right_flux - left_flux)/self.dq1 \
            - (top_flux   - bot_flux )/self.dq2 \
            + self._source(f, self.q1_center, self.q2_center,
                           self.p1, self.p2, self.p3, 
                           self.compute_moments, 
                           self.physical_system.params
                          ) 

    af.eval(df_dt)
    return(df_dt)
 