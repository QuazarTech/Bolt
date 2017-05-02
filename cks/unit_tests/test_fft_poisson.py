import numpy as np
import arrayfire as af
#af.set_backend("cpu")

from cks.poisson_solvers import fft_poisson

def test_1d():
  x   = af.to_array(np.linspace(0, 1, 32))[:-1]
  print(x)
  rho = af.sin(2*np.pi*x)
  dx  = af.sum(x[1] - x[0])

  E_x = fft_poisson(rho, dx)
  assert(abs(af.sum(af.real(E_x) + (0.5/np.pi)*af.cos(2*np.pi*x)))<1e-14)

def test_2d():
  x = af.to_array(np.linspace(0, 1, 32))[:-1]
  y = af.to_array(np.linspace(0, 4, 32))[:-1]
  x = af.tile(x, 1, y.elements())
  y = af.tile(af.reorder(y), x.shape[0], 1)
  rho = af.sin(2*np.pi*x + 4*np.pi*y)

  dx  = af.sum(x[1, 0] - x[0, 0])
  dy  = af.sum(y[0, 1] - y[0, 0])

  E_x, E_y = fft_poisson(rho, dx, dy)

  E_x_ana = -af.cos(2*np.pi*x + 4*np.pi*y) * 2*np.pi/(4*np.pi**2 + 16*np.pi**2)
  E_y_ana = -af.cos(2*np.pi*x + 4*np.pi*y) * 4*np.pi/(4*np.pi**2 + 16*np.pi**2)
  
  error_E_x = abs(af.sum(af.real(E_x) - E_x_ana))
  error_E_y = abs(af.sum(af.real(E_y) - E_y_ana))
  
  assert(error_E_x<1e-14 and error_E_y<1e-14)
