import arrayfire as af

def upwind_flux(left_flux, right_flux, velocity):
    
    flux = af.select(velocity > 0, 
                     left_flux,
                     right_flux
                    )
    
    af.eval(flux)
    return(flux)

def lax_friedrichs_flux(left_flux, right_flux, left_f, right_f, c_lax):
    
    flux = 0.5 * (left_flux + right_flux) - 0.5 * c_lax * (right_f - left_f)

    af.eval(flux)
    return(flux)


def riemann_solver(left_flux, right_flux, left_f, right_f):
    
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

    # bot_flux = af.select(af.broadcast(multiply, f, C_q2) > 0, 
    #                      bot_minus_eps_flux,
    #                      bot_plus_eps_flux
    #                     )

    # top_flux = af.select(af.broadcast(multiply, f, C_q2) > 0, 
    #                      top_minus_eps_flux,
    #                      top_plus_eps_flux,
    #                     )

    return(left_flux, right_flux, bot_flux, top_flux)
