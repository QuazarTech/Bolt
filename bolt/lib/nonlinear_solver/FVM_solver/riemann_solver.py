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


def riemann_solver(self, left_flux, right_flux, left_f, right_f, dim):

    if(self.performance_test_flag == True):    
        tic = af.time()

    if(self.physical_system.params.riemann_solver_in_q == 'upwind-flux'):

        if(self._C_q1.elements() != left_flux.elements()):

            if(dim == 'q1'):
                velocity = af.tile(self._C_q1, 1, 1, left_flux.shape[2], left_flux.shape[3])

            elif(dim == 'q2'):
                velocity = af.tile(self._C_q2, 1, 1, left_flux.shape[2], left_flux.shape[3])

            else:
                raise NotImplementedError('Invalid Option!')

        else:
            if(dim == 'q1'):
                velocity = self._C_q1
            
            elif(dim == 'q2'):
                velocity = self._C_q2

        flux = upwind_flux(left_flux, right_flux, velocity)

    elif(self.physical_system.params.riemann_solver_in_q == 'lax-friedrichs'):

        if(dim == 'q1'):
            c_lax = self.dt/self.dq1 

        elif(dim == 'q2'):
            c_lax = self.dt/self.dq2 

        else:
            raise NotImplementedError('Invalid Option!')

        flux = lax_friedrichs_flux(left_flux, right_flux, left_f, right_f, 
                                   c_lax
                                  )
   
    else:
        raise NotImplementedError('Riemann solver passed is invalid/not-implemented')

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_riemann += toc - tic

    return(flux)
