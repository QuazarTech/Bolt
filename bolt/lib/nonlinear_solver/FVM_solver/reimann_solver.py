import arrayfire as af

from bolt.lib.nonlinear_solver.FVM_solver.reconstruction.minmod \
    import reconstruct_minmod

def riemann_solver(f, C_q1, C_q2):
    
    left_plus_eps_flux, right_minus_eps_flux,\
    bot_plus_eps_flux,  top_minus_eps_flux     = reconstruct_minmod(f, C_q1, C_q2)

    left_minus_eps_flux = af.shift(right_minus_eps_flux, -1)
    right_plus_eps_flux = af.shift(left_plus_eps_flux, 1)
    bot_minus_eps_flux  = af.shift(top_minus_eps_flux, 0, -1)
    top_plus_eps_flux   = af.shift(bot_plus_eps_flux, 1)

    # Upwind fluxes
    left_flux = af.select(f * C_q1 > 0, 
                          left_minus_eps_flux,
                          left_plus_eps_flux
                         )

    right_flux = af.select(f * C_q1 > 0, 
                           right_minus_eps_flux,
                           right_plus_eps_flux,
                          )

    bot_flux = af.select(f * C_q2 > 0, 
                         bot_minus_eps_flux,
                         bot_plus_eps_flux
                        )

    top_flux = af.select(f * C_q2 > 0, 
                         top_minus_eps_flux,
                         top_plus_eps_flux,
                        )

    return(left_flux, right_flux, bot_flux, top_flux)
