from petsc4py import PETSc
import arrayfire as af
import numpy as np

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

  ksp.setTolerances(rtol = 1e-10)
  pde.formRHS(rho, rho_array)
  ksp.solve(rho, phi)

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