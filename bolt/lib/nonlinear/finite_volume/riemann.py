"""
This file contains all the Riemann solvers that are used
by the FVM routines.
"""

import arrayfire as af

def riemann_solver(self, left_state, right_state, velocity):
    """
    Returns the upwinded state, using the 1st order upwind Riemann solver.

    Parameters
    ----------
    left_state : af.Array
                 Array holding the values for the state at the left edge of the cells.

    right_state : af.Array
                 Array holding the values for the state at the right edge of the cells.
    
    velocity : af.Array
               Velocity array whose sign will be used to determine whether the 
               left or right state is chosen.
    """
    if(self.performance_test_flag == True):    
        tic = af.time()

    # Checking if array isn't 4D:
    try:
        size_axis_2 = left_state.shape[2]
    except:
        size_axis_2 = 1

    try:
        size_axis_3 = left_state.shape[3]
    except:
        size_axis_3 = 1

    # Tiling to get to appropriate shape:
    try:
        assert(velocity.shape[2] == left_state.shape[2])
    except:
        velocity = af.tile(velocity, 1, 1, 
                           size_axis_2, size_axis_3
                          )

    upwind_state = af.select(velocity > 0, 
                             left_state,
                             right_state
                            )

    af.eval(upwind_state)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_riemann += toc - tic
    
    return(upwind_state)
