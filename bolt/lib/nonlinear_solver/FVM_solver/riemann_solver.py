import arrayfire as af

def upwind_flux(left_flux, right_flux, velocity):
    
    if(velocity.elements() != left_flux.elements()):
        velocity = af.tile(velocity, 1, left_flux.shape[1], left_flux.shape[2])

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


def riemann_solver(self, left_flux, right_flux, left_f , right_f, dim, method):

    if(self.performance_test_flag == True):    
        tic = af.time()

    if(method == 'upwind-flux'):
        if(dim == 'q1'):
            velocity = self._C_q1
        elif(dim == 'q2'):
            velocity = self._C_q2
        
        elif(dim == 'p1'):
            velocity = self._C_p1
        elif(dim == 'p2'):
            velocity = self._C_p2
        elif(dim == 'p3'):
            velocity = self._C_p3

        flux = upwind_flux(left_flux, right_flux, velocity)

    elif(method == 'lax-friedrichs'):
        flux = lax_friedrichs_flux(left_flux, right_flux,
                                   left_f, right_f, 
                                   self.physical_system.params.c_lax
                                  )
   
    else:
        raise NotImplementedError('Riemann solver passed is invalid/not-implemented')

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_riemann += toc - tic

    return(flux)
