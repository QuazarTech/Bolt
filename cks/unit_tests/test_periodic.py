import numpy as np 
import arrayfire as af 
from cks.boundary_conditions.periodic import periodic_x, periodic_y

class options:
  pass

def test_1d():
  N  = 32
  dx = 1/(N-1)

  config           = options()
  config.N_ghost_x = 3
  config.mode      = '1D1V'

  x = af.to_array(np.linspace(-config.N_ghost_x*dx,\
                               1 + config.N_ghost_x*dx,\
                               N + 2*config.N_ghost_x
                             ))

  x = af.tile(x, 1, N)
  y = af.constant(0, x.shape[0], x.shape[1], dtype = af.Dtype.f64)

  y[config.N_ghost_x:-config.N_ghost_x] = af.sin(2*np.pi*x[config.N_ghost_x:-config.N_ghost_x])

  y = periodic_x(config, y)

  assert(af.max(af.abs(y - af.sin(2*np.pi*x)))<1e-14)

def test_2d_x():
  N_x = 32
  dx  = 1/(N_x-1)

  N_y = 48
  dy  = 1/(N_y-1)

  config           = options()
  config.N_ghost_x = 3
  config.N_ghost_y = 3
  config.mode      = '2D2V'

  x = af.to_array(np.linspace(-config.N_ghost_x*dx,\
                               1 + config.N_ghost_x*dx,\
                               N_x + 2*config.N_ghost_x
                             ))

  y = af.to_array(np.linspace(-config.N_ghost_y*dy,\
                               1 + config.N_ghost_y*dy,\
                               N_y + 2*config.N_ghost_y
                             ))

  x = af.tile(af.reorder(x), N_y + 2*config.N_ghost_y, 1)
  y = af.tile(y, 1, N_x + 2*config.N_ghost_x)

  z = af.constant(0, x.shape[0], x.shape[1], dtype = af.Dtype.f64)
  z[:, config.N_ghost_x:-config.N_ghost_x] = af.sin(2*np.pi*x[:, config.N_ghost_x:-config.N_ghost_x] +\
                                                    4*np.pi*y[:, config.N_ghost_x:-config.N_ghost_x]
                                                   )

  z = periodic_x(config, z)

  assert(af.max(af.abs(z - af.sin(2*np.pi*x + 4*np.pi*y)))<1e-14)

def test_2d_y():
  N_x = 48
  dx  = 1/(N_x-1)

  N_y = 32
  dy  = 1/(N_y-1)

  config           = options()
  config.N_ghost_x = 3
  config.N_ghost_y = 3
  config.mode      = '2D2V'

  x = af.to_array(np.linspace(-config.N_ghost_x*dx,\
                               1 + config.N_ghost_x*dx,\
                               N_x + 2*config.N_ghost_x
                             ))

  y = af.to_array(np.linspace(-config.N_ghost_y*dy,\
                               1 + config.N_ghost_y*dy,\
                               N_y + 2*config.N_ghost_y
                             ))

  x = af.tile(af.reorder(x), N_y + 2*config.N_ghost_y, 1)
  y = af.tile(y, 1, N_x + 2*config.N_ghost_x)

  z = af.constant(0, x.shape[0], x.shape[1], dtype = af.Dtype.f64)
  z[config.N_ghost_y:-config.N_ghost_y, :] = af.sin(2*np.pi*x[config.N_ghost_y:-config.N_ghost_y, :] +\
                                                    4*np.pi*y[config.N_ghost_y:-config.N_ghost_y, :]
                                                   )

  z = periodic_y(config, z)

  assert(af.max(af.abs(z - af.sin(2*np.pi*x + 4*np.pi*y)))<1e-14)
