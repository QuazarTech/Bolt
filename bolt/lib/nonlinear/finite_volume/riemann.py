"""
This file contains all the Riemann solvers that are used
by the FVM routines.
"""

import arrayfire as af

def upwind_flux(left_flux, right_flux, velocity):
    """
    Returns the flux, using the 1st order upwind flux Riemann solver.

    Parameters
    ----------

    left_flux : af.Array
                Array holding the values for the flux at the left edge of the cells.

    right_flux : af.Array
                Array holding the values for the flux at the right edge of the cells.
    
    velocity : af.Array
               Velocity array whose sign will be used to determine whether the 
               left or right flux is chosen.
    """
    flux = af.select(velocity > 0, 
                     left_flux,
                     right_flux
                    )
    
    af.eval(flux)
    return(flux)

def lax_friedrichs_flux(left_flux, right_flux, left_f, right_f, c_lax):
    """
    Returns the flux, using the Local Lax Friedrichs Riemann solver.
    **NOT TESTED**
    Parameters
    ----------

    left_flux : af.Array
                Array holding the values for the flux at the left edge of the cells.

    right_flux : af.Array
                 Array holding the values for the flux at the right edge of the cells.

    left_f : af.Array
             Array holding the values for the distribution function at the left edge 
             of the cells.

    right_f : af.Array
              Array holding the values for the distribution function  at the right 
              edge of the cells.
    
    c_lax : double
            c_lax which it to be used.
    """
    
    flux = 0.5 * (left_flux + right_flux) - 0.5 * c_lax * (right_f - left_f)

    af.eval(flux)
    return(flux)


def riemann_solver(self, left_flux, right_flux, left_f, right_f, method, dim):
    """
    Calls the appropriate Riemann solver as defined by the user.

    Parameters
    ----------

    left_flux : af.Array
                Array holding the values for the flux at the left edge of the cells.

    right_flux : af.Array
                 Array holding the values for the flux at the right edge of the cells.

    left_f : af.Array
             Array holding the values for the distribution function at the left edge 
             of the cells.

    right_f : af.Array
              Array holding the values for the distribution function  at the right 
              edge of the cells.

    method : str
             Riemann solver method which is to be used.

    dim: str
         Dimension along which the Riemann solver is to be applied.
    
    """
    if(self.performance_test_flag == True):    
        tic = af.time()

    if(method == 'upwind-flux'):

        if(dim == 'q1' or dim == 'q2'):

            if(self._C_q1.elements() != left_flux.elements()):

                if(dim == 'q1'):
                    velocity = af.tile(self._C_q1, 1, 1, 
                                       left_flux.shape[2], left_flux.shape[3]
                                      )

                elif(dim == 'q2'):
                    velocity = af.tile(self._C_q2, 1, 1, 
                                       left_flux.shape[2], left_flux.shape[3]
                                      )
    
            else:
                
                if(dim == 'q1'):
                    velocity = self._C_q1
                
                elif(dim == 'q2'):
                    velocity = self._C_q2

        else:

            if(dim == 'p1'):
                velocity = self._C_p1
            
            elif(dim == 'p2'):
                velocity = self._C_p2

            elif(dim == 'p3'):
                velocity = self._C_p3

            else:
                raise NotImplementedError('Invalid Option!')
        
        flux = upwind_flux(left_flux, right_flux, velocity)

    elif(method == 'lax-friedrichs'):

        if(dim == 'q1'):
            c_lax = self.dt/self.dq1 

        elif(dim == 'q2'):
            c_lax = self.dt/self.dq2 

        elif(dim == 'p1'):
            c_lax = self.dt/self.dp1 

        elif(dim == 'p2'):
            c_lax = self.dt/self.dp2 

        elif(dim == 'p3'):
            c_lax = self.dt/self.dp3

        else:
            raise NotImplementedError('Invalid Dimension!')

        flux = lax_friedrichs_flux(left_flux, right_flux, 
                                   left_f, right_f, 
                                   c_lax
                                  )
   
    else:
        raise NotImplementedError('Riemann solver passed is invalid/not-implemented')

    af.eval(flux)
    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_riemann += toc - tic

    return(flux)
