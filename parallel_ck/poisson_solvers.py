import arrayfire as af
from scipy.fftpack import fftfreq
import numpy as np

def fft_poisson(rho, dx, dy = None):
  
  k_x = af.to_array(fftfreq(rho.shape[1], dx))
  k_x = af.Array.as_type(k_x, af.Dtype.c64)
  k_y = af.to_array(fftfreq(rho.shape[0], dy))
  k_x = af.tile(af.reorder(k_x), rho.shape[0], 1)
  k_y = af.tile(k_y, 1, rho.shape[1])
  k_y = af.Array.as_type(k_y, af.Dtype.c64)

  rho_hat       = af.fft2(rho)
  potential_hat = af.constant(0, rho.shape[0], rho.shape[1], dtype=af.Dtype.c64)
  
  potential_hat       = (1/(4 * np.pi**2 * (k_x*k_x + k_y*k_y))) * rho_hat
  potential_hat[0, 0] = 0
  
  E_x_hat = -1j * 2 * np.pi * (k_x) * potential_hat
  E_y_hat = -1j * 2 * np.pi * (k_y) * potential_hat

  E_x = af.ifft2(E_x_hat)
  E_y = af.ifft2(E_y_hat)

  af.eval(E_x, E_y)
  return(E_x, E_y)