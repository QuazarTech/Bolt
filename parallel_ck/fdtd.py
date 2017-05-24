from cks.evolve import communicate_fields
import arrayfire as af

def fdtd(da, config, E_x, E_y, E_z, B_x, B_y, B_z, J_x, J_y, J_z, dt):
    
  # E's and B's are staggered in time such that
  # B's are defined at (n + 1/2), and E's are defined at n 
  
  # Positions of grid point where field quantities are defined:
  # B_x --> (i, j + 1/2)
  # B_y --> (i + 1/2, j)
  # B_z --> (i + 1/2, j + 1/2)
  
  # E_x --> (i + 1/2, j)
  # E_y --> (i, j + 1/2)
  # E_z --> (i, j)
  
  # J_x --> (i + 1/2, j)
  # J_y --> (i, j + 1/2)
  # J_z --> (i, j)

  N_x = config.N_x
  N_y = config.N_y

  x_start = config.x_start
  x_end   = config.x_end

  y_start = config.y_start
  y_end   = config.y_end

  dx = (x_end - x_start)/(N_x - 1)
  dy = (y_end - y_start)/(N_y - 1)

  glob  = da.createGlobalVec()
  local = da.createLocalVec()

  B_x = communicate_fields(da, config, B_x, local, glob)
  B_y = communicate_fields(da, config, B_y, local, glob)
  B_z = communicate_fields(da, config, B_z, local, glob)
  
  E_x = communicate_fields(da, config, E_x, local, glob)
  E_y = communicate_fields(da, config, E_y, local, glob)
  E_z = communicate_fields(da, config, E_z, local, glob)

  E_x +=   (dt/dy) * (B_z - af.shift(B_z, 1, 0)) - J_x * dt
  E_y +=  -(dt/dx) * (B_z - af.shift(B_z, 0, 1)) - J_y * dt
  E_z +=   (dt/dx) * (B_y - af.shift(B_y, 0, 1)) \
         - (dt/dy) * (B_x - af.shift(B_x, 0, 1)) \
         - dt * J_z
          
  E_x = communicate_fields(da, config, E_x, local, glob)
  E_y = communicate_fields(da, config, E_y, local, glob)
  E_z = communicate_fields(da, config, E_z, local, glob)

  B_x +=  -(dt/dy) * (af.shift(E_z, -1, 0) - E_z)
  B_y +=   (dt/dx) * (af.shift(E_z, 0, -1) - E_z)
  B_z += - (dt/dx) * (af.shift(E_y, 0, -1) - E_y) \
         + (dt/dy) * (af.shift(E_x, -1, 0) - E_x)
      
  B_x = communicate_fields(da, config, B_x, local, glob)
  B_y = communicate_fields(da, config, B_y, local, glob)
  B_z = communicate_fields(da, config, B_z, local, glob)

  af.eval(E_x, E_y, E_z, B_x, B_y, B_z)
  return(E_x, E_y, E_z, B_x, B_y, B_z)

def fdtd_grid_to_ck_grid(da, config, E_x, E_y, E_z, B_x, B_y, B_z):

  glob  = da.createGlobalVec()
  local = da.createLocalVec()
    
  E_x = 0.5 * (E_x + af.shift(E_x, 0, 1))
  B_x = 0.5 * (B_x + af.shift(B_x, 1, 0))

  E_y = 0.5 * (E_y + af.shift(E_y, 1, 0))
  B_y = 0.5 * (B_y + af.shift(B_y, 0, 1))

  B_z = 0.25 * (
                B_y + af.shift(B_z, 0, 1) + \
                af.shift(B_z, 1, 0) + af.shift(B_z, 1, 1)
               )

  B_x = communicate_fields(da, config, B_x, local, glob)
  B_y = communicate_fields(da, config, B_y, local, glob)
  B_z = communicate_fields(da, config, B_z, local, glob)
  
  E_x = communicate_fields(da, config, E_x, local, glob)
  E_y = communicate_fields(da, config, E_y, local, glob)
  E_z = communicate_fields(da, config, E_z, local, glob)

  af.eval(E_x, E_y, E_z, B_x, B_y, B_z)
  return(E_x, E_y, E_z, B_x, B_y, B_z)

def ck_grid_to_fdtd_grid(config, E_x, E_y, E_z, B_x, B_y, B_z):

  E_x = 0.5 * (E_x + af.shift(E_x, 0, -1))
  B_x = 0.5 * (B_x + af.shift(B_x, -1, 0))

  E_y = 0.5 * (E_y + af.shift(E_y, -1, 0))
  B_y = 0.5 * (B_y + af.shift(B_y, 0, -1))

  B_z = 0.25 * (
                B_y + af.shift(B_z, 0, -1) + \
                af.shift(B_z, -1, 0) + af.shift(B_z, -1, -1)
               )


  E_x = periodic_x(config, E_x)
  E_y = periodic_x(config, E_y)

  B_x = periodic_x(config, B_x)
  B_y = periodic_x(config, B_y)
  B_z = periodic_x(config, B_z)

  E_x = periodic_y(config, E_x)
  E_y = periodic_y(config, E_y)

  B_x = periodic_y(config, B_x)
  B_y = periodic_y(config, B_y)
  B_z = periodic_y(config, B_z)

  af.eval(E_x, E_y, E_z, B_x, B_y, B_z)
  return(E_x, E_y, E_z, B_x, B_y, B_z)