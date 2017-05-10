from cks.boundary_conditions.periodic import periodic_x, periodic_y
import arrayfire as af

def mode1_fdtd(config, E_z, B_x, B_y, J_z, dt):

  N_x = config.N_x
  N_y = config.N_y

  left_boundary  = config.left_boundary
  right_boundary = config.right_boundary
  length_x       = right_boundary - left_boundary

  bot_boundary = config.bot_boundary
  top_boundary = config.top_boundary
  length_y     = top_boundary - bot_boundary

  """Enforcing BC's"""

  E_z = periodic_x(config, E_z)
  B_x = periodic_x(config, B_x)
  B_y = periodic_x(config, B_y)

  E_z = periodic_y(config, E_z)
  B_x = periodic_y(config, B_x)
  B_y = periodic_y(config, B_y)

  """ Setting division size"""

  dx = length_x/(N_x - 1)
  dy = length_y/(N_y - 1)

  """  Updating the Magnetic fields   """
  
  E_z +=   (dt/dx) * (af.signal.convolve2_separable(af.Array([0, 1, 0]),\
                                                    af.Array([0, 1, -1]), B_y)) \
         - (dt/dy) * (af.signal.convolve2_separable(af.Array([0, 1, -1]),\
                                                    af.Array([0, 1, 0]), B_x)) \
         - dt * J_z

  E_z = periodic_x(config, E_z)
  E_z = periodic_y(config, E_z)

  B_x += -(dt/dy)*(af.signal.convolve2_separable(af.Array([1, -1, 0]),\
                                                 af.Array([0, 1, 0]), E_z))

  B_y += (dt/dx)*(af.signal.convolve2_separable(af.Array([0, 1, 0]),\
                                                af.Array([1, -1, 0]), E_z))

  B_x = periodic_x(config, B_x)
  B_y = periodic_x(config, B_y)

  B_x = periodic_y(config, B_x)
  B_y = periodic_y(config, B_y)



  af.eval(E_z, B_x, B_y)
  return(E_z, B_x, B_y)

def mode2_fdtd(config, B_z, E_x, E_y, J_x, J_y, dt):

  N_x = config.N_x
  N_y = config.N_y

  left_boundary  = config.left_boundary
  right_boundary = config.right_boundary
  length_x       = right_boundary - left_boundary

  bot_boundary = config.bot_boundary
  top_boundary = config.top_boundary
  length_y     = top_boundary - bot_boundary

  """Enforcing BC's"""

  B_z = periodic_x(config, B_z)
  E_x = periodic_x(config, E_x)
  E_y = periodic_x(config, E_y)

  B_z = periodic_y(config, B_z)
  E_x = periodic_y(config, E_x)
  E_y = periodic_y(config, E_y)

  """ Setting division size"""

  dx = length_x/(N_x - 1)
  dy = length_y/(N_y - 1)

  """  Updating the Magnetic field  """
  E_x += (dt/dy)  * (af.signal.convolve2_separable(af.Array([0, 1, -1]),\
                                                   af.Array([0, 1, 0]), B_z)) - J_x * dt

  E_y += -(dt/dx) * (af.signal.convolve2_separable(af.Array([0, 1, 0]), \
                                                   af.Array([0, 1, -1]), B_z)) - J_y * dt


  E_x = periodic_x(config, E_x)
  E_y = periodic_x(config, E_y)

  E_x = periodic_y(config, E_x)
  E_y = periodic_y(config, E_y)
  
  B_z += - (dt/dx) * (af.signal.convolve2_separable(af.Array([0,  1, 0]),\
                                                    af.Array([1, -1, 0]), E_y)) \
         + (dt/dy) * (af.signal.convolve2_separable(af.Array([1, -1, 0]),\
                                                    af.Array([0, 1, 0]), E_x))
  
  B_z = periodic_x(config, B_z)
  B_z = periodic_y(config, B_z)

  af.eval(B_z, E_x, E_y)
  return(B_z, E_x, E_y)

def fdtd(config, E_x, E_y, E_z, B_x, B_y, B_z, J_x, J_y, J_z, dt):

  E_z_new, B_x_new, B_y_new = mode1_fdtd(config, E_z, B_x, B_y, J_z, dt)
  B_z_new, E_x_new, E_y_new = mode2_fdtd(config, B_z, E_x, E_y, J_x, J_y, dt)
  
  af.eval(E_x_new, E_y_new, E_z_new, B_x_new, B_y_new, B_z_new)
  return (E_x_new, E_y_new, E_z_new, B_x_new, B_y_new, B_z_new)

def fdtd_grid_to_ck_grid(config, E_x, E_y, E_z, B_x, B_y, B_z):
    
  E_z = 0.5 * (E_z + af.shift(E_z, -1, 0))
  B_z = 0.5 * (B_z + af.shift(B_z, 0, 1))

  E_x = 0.25 * (E_x + af.shift(E_x, -1, 0) + af.shift(E_x, 0, 1) + af.shift(E_x, -1, 1))
  B_y = 0.25 * (B_y + af.shift(B_y, -1, 0) + af.shift(B_y, 0, 1) + af.shift(B_y, -1, 1))

  E_x_ck = periodic_x(config, E_x)
  E_y_ck = periodic_x(config, E_y)
  E_z_ck = periodic_x(config, E_z)

  B_x_ck = periodic_x(config, B_x)
  B_y_ck = periodic_x(config, B_y)
  B_z_ck = periodic_x(config, B_z)

  E_x_ck = periodic_y(config, E_x)
  E_y_ck = periodic_y(config, E_y)
  E_z_ck = periodic_y(config, E_z)

  B_x_ck = periodic_y(config, B_x)
  B_y_ck = periodic_y(config, B_y)
  B_z_ck = periodic_y(config, B_z)

  af.eval(E_x_ck, E_y_ck, E_z_ck, B_x_ck, B_y_ck, B_z_ck)
  return(E_x_ck, E_y_ck, E_z_ck, B_x_ck, B_y_ck, B_z_ck)

def ck_grid_to_fdtd_grid(config, E_x, E_y, E_z, B_x, B_y, B_z):

  E_z = 0.5 * (E_z + af.shift(E_z, 1, 0))
  B_z = 0.5 * (B_z + af.shift(B_z, 0, -1))

  E_x = 0.25 * (E_x + af.shift(E_x, 0, -1) + af.shift(E_x, 1, 0) + af.shift(E_x, 1, -1))
  B_y = 0.25 * (B_y + af.shift(B_y, 0, -1) + af.shift(B_y, 1, 0) + af.shift(B_y, 1, -1))

  E_x_fdtd = periodic_x(config, E_x)
  E_y_fdtd = periodic_x(config, E_y)
  E_z_fdtd = periodic_x(config, E_z)

  B_x_fdtd = periodic_x(config, B_x)
  B_y_fdtd = periodic_x(config, B_y)
  B_z_fdtd = periodic_x(config, B_z)

  E_x_fdtd = periodic_y(config, E_x)
  E_y_fdtd = periodic_y(config, E_y)
  E_z_fdtd = periodic_y(config, E_z)

  B_x_fdtd = periodic_y(config, B_x)
  B_y_fdtd = periodic_y(config, B_y)
  B_z_fdtd = periodic_y(config, B_z)

  af.eval(E_x_fdtd, E_y_fdtd, E_z_fdtd, B_x_fdtd, B_y_fdtd, B_z_fdtd)
  return(E_x_fdtd, E_y_fdtd, E_z_fdtd, B_x_fdtd, B_y_fdtd, B_z_fdtd)