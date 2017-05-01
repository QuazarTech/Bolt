"""
This file contains the various solver for the Poisson Equation:
d^2V/dx^2 + d^2V/dy^2 = -rho
It returns the value of electric field in the x and y directions,
which will be used in solving the problem setup.
""" 
import arrayfire as af
from scipy.fftpack import fftfreq
import numpy as np

def fft_poisson(rho, dx, dy = None):
  """
  FFT solver which returns the value of electric field. This will only work
  when the system being solved for has periodic boundary conditions.

  Parameters:
  -----------
    rho : The 1D/2D density array obtained from calculate_density() is passed 
          to this function.

    dx  : Step size in the x-grid

    dy  : Step size in the y-grid. Set to None by default to avoid conflicts 
          with the 1D case.

  Output:
  -------
    E_x, E_y : Depending on the dimensionality of the system considered, either both 
               E_x, and E_y are returned or E_x is returned.
  """

  if(len(rho.shape) == 2):
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

  else:
    k_x = af.to_array(fftfreq(rho.shape[0], dx))
    k_x = af.Array.as_type(k_x, af.Dtype.c64)

    rho_hat       = af.fft(rho)
    potential_hat = af.constant(0, af.Array.elements(rho), dtype = af.Dtype.c64)
    
    potential_hat[1:] =  (1/(4 * np.pi**2 * k_x[1:]**2)) * rho_hat[1:]
    potential_hat[0]  =  0

    E_x_hat =  -1j * 2 * np.pi * k_x * potential_hat
    E_x     = af.ifft(E_x_hat)
    
    af.eval(E_x)
    return E_x