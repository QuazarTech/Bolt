import arrayfire as af

from bolt.lib.nonlinear_solver.FVM_solver.reconstruction.minmod \
    import reconstructMM

def riemann_solver():
    # TODO
    # WHAT TO ADD HERE?
    # Unable to understand from 
    # https://github.com/mchandra/grim/blob/master/src/physics/riemannsolver.cpp

def upwind_flux(self):
    left, right, bot, top = reconstructMM(self)

    # Upwind fluxes
    left_flux = af.select(self.f * self.C_q1 > 0, 
                          af.shift(right, -1),
                          left
                         )

    right_flux = af.select(self.f * self.C_q1 > 0, 
                           right,
                           af.shift(left, 1),
                          )

    bot_flux = af.select(self.f * self.C_q2 > 0, 
                         af.shift(top, 0, -1),
                         bot
                        )

    top_flux = af.select(self.f * self.C_q2 > 0, 
                         top,
                         af.shift(bot, 1),
                        )

    return(left_flux, right_flux, bot_flux, top_flux)
