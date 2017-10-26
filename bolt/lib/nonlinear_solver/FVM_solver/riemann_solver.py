import arrayfire as af

from .reconstruction.minmod import reconstruct_minmod
from .reconstruction.ppm import reconstruct_ppm
from .reconstruction.weno5 import reconstruct_weno5

def riemann_solver(f, C_q1, C_q2):
    
    # left_plus_eps_flux, right_minus_eps_flux,\
    # bot_plus_eps_flux,  top_minus_eps_flux     = reconstruct_ppm(f, C_q1, C_q2)
    multiply = lambda a, b: a * b

    left_plus_eps_flux   = af.broadcast(multiply, C_q1, f)
    right_minus_eps_flux = af.broadcast(multiply, C_q1, f)
    bot_plus_eps_flux    = af.broadcast(multiply, C_q2, f)
    top_minus_eps_flux   = af.broadcast(multiply, C_q2, f)

    left_minus_eps_flux = af.shift(right_minus_eps_flux, -1)
    right_plus_eps_flux = af.shift(left_plus_eps_flux, 1)
    bot_minus_eps_flux  = af.shift(top_minus_eps_flux, 0, -1)
    top_plus_eps_flux   = af.shift(bot_plus_eps_flux, 1)

    # left_minus_eps_flux = af.shift(right_minus_eps_flux, -1)
    # right_plus_eps_flux = af.shift(left_plus_eps_flux, 1)
    # bot_minus_eps_flux  = af.shift(top_minus_eps_flux, 0, -1)
    # top_plus_eps_flux   = af.shift(bot_plus_eps_flux, 1)

    left_flux = 0.5 * (left_minus_eps_flux + left_plus_eps_flux)
    right_flux = 0.5 * (right_minus_eps_flux + right_plus_eps_flux)
    bot_flux = 0.5 * (bot_minus_eps_flux + bot_plus_eps_flux)
    top_flux = 0.5 * (top_minus_eps_flux + top_plus_eps_flux)

    # Upwind fluxes
    # left_flux = af.select(af.broadcast(multiply, f, C_q1) > 0, 
    #                       left_minus_eps_flux,
    #                       left_plus_eps_flux
    #                      )

    # right_flux = af.select(af.broadcast(multiply, f, C_q1) > 0, 
    #                        right_minus_eps_flux,
    #                        right_plus_eps_flux,
    #                       )

    # bot_flux = af.select(af.broadcast(multiply, f, C_q2) > 0, 
    #                      bot_minus_eps_flux,
    #                      bot_plus_eps_flux
    #                     )

    # top_flux = af.select(af.broadcast(multiply, f, C_q2) > 0, 
    #                      top_minus_eps_flux,
    #                      top_plus_eps_flux,
    #                     )

    return(left_flux, right_flux, bot_flux, top_flux)
