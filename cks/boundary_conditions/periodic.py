"""
This module contains the functions that impose periodic boundary
conditions in the x and y directions. The functions are 
appropriately named periodic_x and periodic_y, and return the
spatial quantities after imposing periodic boundary conditions
"""
import arrayfire as af

def periodic_x(config, f):
  """
  Applies periodic boundary conditions in x to the spatial quantity that
  is passed as an argument.

  Parameters:
  -----------
    config: Object config which is obtained by set() is passed to this file

    f : 2D/4D array that is passed to this function. This array can be anything
        upon which periodic boundary conditions need to be enforced such as
        electric field, density etc. and is not restricted to the distribution 
        function only.

  Output:
  -------
    f : Returns the passed array after applying periodic boundary 
        conditions.
  """
  
  N_ghost_x = config.N_ghost_x

  if(config.mode == '2D2V'):
    f[:, :N_ghost_x]  = f[:, -(2*N_ghost_x + 1):-(N_ghost_x + 1)]
    f[:, -N_ghost_x:] = f[:, (N_ghost_x + 1):(2*N_ghost_x + 1)]
    
  else:
    f[:N_ghost_x]  = f[-(2*N_ghost_x + 1):-(N_ghost_x + 1)]
    f[-N_ghost_x:] = f[(N_ghost_x + 1):(2*N_ghost_x + 1)]  

  af.eval(f)
  return(f)

def periodic_y(config, f):
  """
  Applies periodic boundary conditions in y to the spatial quantity that
  is passed as an argument.

  Parameters:
  -----------
    config: Object config which is obtained by set() is passed to this file

    f : 2D/4D array that is passed to this function. This array can be anything
        upon which periodic boundary conditions need to be enforced such as
        electric field, density etc. and is not restricted to the distribution 
        function only.

  Output:
  -------
    f : Returns the passed array after applying periodic boundary 
        conditions.
  """

  if(config.mode != '2D2V'):
    raise Exception('Not in 2D mode!')

  N_ghost_y = config.N_ghost_y

  f[:N_ghost_y]  = f[-(2*N_ghost_y + 1):-(N_ghost_y + 1)]
  f[-N_ghost_y:] = f[(N_ghost_y + 1):(2*N_ghost_y + 1)]
    
  af.eval(f)
  return(f)