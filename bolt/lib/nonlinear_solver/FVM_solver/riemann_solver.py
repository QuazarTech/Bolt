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


def riemann_solver(self, left_flux, right_flux, left_f, right_f):

    if(self.performance_test_flag == True):    
        tic = af.time()

    if(self.physical_system.params.riemann_solver == 'upwind-flux'):
        velocity = af.tile(self._C_q1, 1, left_flux.shape[1], left_flux.shape[2])
        flux     = upwind_flux(left_flux, right_flux, velocity)

    elif(self.physical_system.params.riemann_solver == 'lax-friedrichs'):
        flux = lax_friedrichs_flux(left_flux, right_flux, left_f, right_f, 0.016)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_riemann += toc - tic

    return(flux)
