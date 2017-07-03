from petsc4py import PETSc
import arrayfire as af
import numpy as np
from scipy.fftpack import fftfreq

# This user class is an application context for the problem at hand; 
# It contains some parametes and frames the matrix system depending on the system state
class Poisson2D(object):

  def __init__(self, da, config):
    assert da.getDim() == 2
    self.da     = da
    self.config = config
    self.localX = da.createLocalVec()

  def formRHS(self, rho, rho_array):
    rho_val = self.da.getVecArray(rho)
      
    dx = (self.config.x_end - self.config.x_start)/self.config.N_x
    dy = (self.config.y_end - self.config.y_start)/self.config.N_y

    rho_val[:] = rho_array * dx * dy

  def mult(self, mat, X, Y):
        
    self.da.globalToLocal(X, self.localX)
    
    x = self.da.getVecArray(self.localX)
    y = self.da.getVecArray(Y)
    
    dx = (self.config.x_end - self.config.x_start)/self.config.N_x
    dy = (self.config.y_end - self.config.y_start)/self.config.N_y

    (y_start, y_end), (x_start, x_end) = self.da.getRanges()
    
    for j in range(y_start, y_end):
      for i in range(x_start, x_end):
        u    = x[j, i]   # center
        u_w  = x[j, i-1] # west
        u_e  = x[j, i+1] # east
        u_s  = x[j-1, i] # south
        u_n  = x[j+1, i] # north
        
        u_xx = (-u_e + 2*u - u_w)*dy/dx
        u_yy = (-u_n + 2*u - u_s)*dx/dy
 
        y[j, i] = u_xx + u_yy

def solve_electrostatic_fields(da, config, rho_array):
  dx = (config.x_end - config.x_start)/config.N_x
  dy = (config.y_end - config.y_start)/config.N_y  

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  pde = Poisson2D(da, config)
  phi = da.createGlobalVec()
  rho = da.createGlobalVec()

  phi_local = da.createLocalVec()

  A = PETSc.Mat().createPython([phi.getSizes(), rho.getSizes()], comm = da.comm)
  A.setPythonContext(pde)
  A.setUp()

  ksp = PETSc.KSP().create()

  ksp.setOperators(A)
  ksp.setType('cg')

  pc = ksp.getPC()
  pc.setType('none')

  ksp.setTolerances(atol = 1e-5)
  pde.formRHS(rho, rho_array)
  ksp.solve(rho, phi)

  print(ksp.converged)
  
  if(ksp.converged != True):
    raise Exception
  
  da.globalToLocal(phi, phi_local)

  # Since rho was defined at (i + 0.5, j + 0.5) 
  # Electric Potential returned will also be at (i + 0.5, j + 0.5)
  electric_potential = af.to_array(np.swapaxes(phi_local[:].reshape(N_x_local + 2*config.N_ghost,\
                                                                    N_y_local + 2*config.N_ghost
                                                                   ), 0, 1
                                              )
                                  )

  E_x = -(af.shift(electric_potential, 0, -1) - electric_potential)/dx #(i, j+1/2)
  E_y = -(af.shift(electric_potential, -1, 0) - electric_potential)/dy #(i+1/2, j)

  # Obtaining the values at (i+0.5, j+0.5):
  E_x = 0.5 * (E_x + af.shift(E_x, 0, -1))
  E_y = 0.5 * (E_y + af.shift(E_y, -1, 0))

  af.eval(E_x, E_y)
  return(E_x, E_y)

def fft_poisson(rho, dx, dy):
  """
  FFT solver which returns the value of electric field. This will only work
  when the system being solved for has periodic boundary conditions.
  Parameters:
  -----------
    rho : The 1D/2D density array obtained from calculate_density() is passed to this
          function.
    dx  : Step size in the x-grid
    dy  : Step size in the y-grid
  Output:
  -------
    E_x, E_y : Depending on the dimensionality of the system considered, either both E_x, and
               E_y are returned or E_x is returned.
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

    E_x = af.real(af.ifft2(E_x_hat))
    E_y = af.real(af.ifft2(E_y_hat))

    af.eval(E_x, E_y)
    return(E_x, E_y)