import numpy as np
import arrayfire as af

from cks.compute_moments import calculate_density, calculate_vel_bulk_x,\
                                calculate_vel_bulk_y, calculate_vel_bulk_z,\
                                calculate_mom_bulk_x, calculate_mom_bulk_y,\
                                calculate_mom_bulk_z, calculate_temperature

from cks.interpolation_routines import f_interp_vel_3d
from petsc4py import PETSc

def communicate_distribution_function(da, args, local, glob):

  # Accessing the values of the global and local Vectors
  local_value = da.getVecArray(local)
  glob_value  = da.getVecArray(glob)

  N_ghost = args.config.N_ghost

  # Storing values of af.Array in PETSc.Vec:
  local_value[:] = np.array(args.f)
  
  # Global value is non-inclusive of the ghost-zones:
  glob_value[:] = (local_value[:])[N_ghost:-N_ghost,\
                                   N_ghost:-N_ghost,\
                                   :
                                  ]

  # The following function takes care of the boundary conditions, 
  # and interzonal communications:
  da.globalToLocal(glob, local)

  # Converting back from PETSc.Vec to af.Array:
  f_updated = af.to_array(local_value[:])

  return(f_updated)

def communicate_fields(da, config, local_field, local, glob):

  # Accessing the values of the global and local Vectors
  local_value = da.getVecArray(local)
  glob_value  = da.getVecArray(glob)

  N_ghost = config.N_ghost

  local_value[:] = np.array(local_field)
  
  # Global value is non-inclusive of the ghost-zones:
  glob_value[:] = (local_value[:])[N_ghost:-N_ghost,\
                                   N_ghost:-N_ghost
                                  ]

  # Takes care of boundary conditions and interzonal communications:
  da.globalToLocal(glob, local)

  # Converting back to af.Array
  field_updated = af.to_array(local_value[:])

  af.eval(field_updated)
  return(field_updated)

def f_MB(da, args):
  # All the arrays need to be in velocitiesExpanded form
  config = args.config
  f      = args.f

  vel_x = args.vel_x
  vel_y = args.vel_y
  vel_z = args.vel_z

  dv_x = (2*config.vel_x_max)/config.N_vel_x
  dv_y = (2*config.vel_y_max)/config.N_vel_y
  dv_z = (2*config.vel_z_max)/config.N_vel_z
  
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  vel_bulk_x_background  = config.vel_bulk_x_background
  vel_bulk_y_background  = config.vel_bulk_y_background
  vel_bulk_z_background  = config.vel_bulk_z_background

  if(config.mode == '3V'):
    n          = af.tile(calculate_density(args), 1, f.shape[1], f.shape[2], f.shape[3])
    T          = af.tile(calculate_temperature(args), 1, f.shape[1], f.shape[2], f.shape[3])
    vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, f.shape[1], f.shape[2], f.shape[3])
    vel_bulk_y = af.tile(calculate_vel_bulk_y(args), 1, f.shape[1], f.shape[2], f.shape[3])
    vel_bulk_z = af.tile(calculate_vel_bulk_z(args), 1, f.shape[1], f.shape[2], f.shape[3])
    
    f_MB = n * (mass_particle/(2*np.pi*boltzmann_constant*T))**(3/2) * \
           af.exp(-mass_particle*(vel_x - vel_bulk_x)**2/(2*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_y - vel_bulk_y)**2/(2*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_z - vel_bulk_z)**2/(2*boltzmann_constant*T))

    f_background = rho_background * \
                   (mass_particle/(2*np.pi*boltzmann_constant*temperature_background))**(3/2) * \
                    af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                          (2*boltzmann_constant*temperature_background)) * \
                    af.exp(-mass_particle*(vel_y - vel_bulk_y_background)**2/\
                          (2*boltzmann_constant*temperature_background)) * \
                    af.exp(-mass_particle*(vel_z - vel_bulk_z_background)**2/\
                          (2*boltzmann_constant*temperature_background))

  elif(config.mode == '2V'):
    n          = af.tile(calculate_density(args), 1, f.shape[1], f.shape[2], 1)
    T          = af.tile(calculate_temperature(args), 1, f.shape[1], f.shape[2], 1)
    vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, f.shape[1], f.shape[2], 1)
    vel_bulk_y = af.tile(calculate_vel_bulk_y(args), 1, f.shape[1], f.shape[2], 1)
    
    f_MB = n * (mass_particle/(2*np.pi*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_x - vel_bulk_x)**2/(2*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_y - vel_bulk_y)**2/(2*boltzmann_constant*T))

    f_background = rho_background * \
                   (mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                    af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                          (2*boltzmann_constant*temperature_background)) * \
                    af.exp(-mass_particle*(vel_y - vel_bulk_y_background)**2/\
                          (2*boltzmann_constant*temperature_background))
  else:
    n          = af.tile(calculate_density(args), 1, f.shape[1], f.shape[2], 1)
    T          = af.tile(calculate_temperature(args), 1, f.shape[1], f.shape[2], 1)
    vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, f.shape[1], f.shape[2], 1)

    f_MB = n*af.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T))*\
             af.exp(-mass_particle*(vel_x-vel_bulk_x)**2/(2*boltzmann_constant*T))

    f_background = rho_background * \
                   np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                   af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                         (2*boltzmann_constant*temperature_background))

  normalization = af.sum(f_background)*dv_x*dv_y*dv_z/(f.shape[0])
  f_MB          = f_MB/normalization

  af.eval(f_MB)
  return(f_MB)

def collision_step(da, args, dt):

  tau     = args.config.tau
  N_ghost = args.config.N_ghost
  N_vel_x = args.config.N_vel_x
  N_vel_y = args.config.N_vel_y
  N_vel_z = args.config.N_vel_z
  
  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  # Converting from positionsExpanded form to velocitiesExpanded form:
  args.f = af.moddims(args.f,                   
                      (N_x_local + 2 * N_ghost)*\
                      (N_y_local + 2 * N_ghost),\
                      N_vel_y,\
                      N_vel_x,\
                      N_vel_z
                     ) 

  # Performing the step of df/dt = C[f] = -(f - f_MB)/tau:
  f0             = f_MB(da, args)
  f_intermediate = args.f - (dt/2)*(args.f - f0)/tau
  args.f         = args.f - (dt)  *(f_intermediate - f0)/tau

  # Converting from velocitiesExpanded form to positionsExpanded form:
  args.f = af.moddims(args.f,                   
                      (N_y_local + 2 * N_ghost),\
                      (N_x_local + 2 * N_ghost),\
                      N_vel_y*N_vel_x*N_vel_z,\
                      1
                     )
  af.eval(args.f)
  return(args.f)

class Poisson2D(object):

  def __init__(self, da, config):
    assert da.getDim() == 2
    self.da     = da
    self.config = config
    self.localX = da.createLocalVec()

  def formRHS(self, rho, rho_array):
    rho_val = self.da.getVecArray(rho)
      
    dx = (self.config.x_start - self.config.x_end)/self.config.N_x
    dy = (self.config.y_start - self.config.y_end)/self.config.N_y

    rho_val[:] = rho_array * dx * dy
        
  def mult(self, mat, X, Y):
        
    self.da.globalToLocal(X, self.localX)
    
    x = self.da.getVecArray(self.localX)
    y = self.da.getVecArray(Y)
    
    dx = (self.config.x_start - self.config.x_end)/self.config.N_x
    dy = (self.config.y_start - self.config.y_end)/self.config.N_y
    
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
  dx = (config.x_start - config.x_end)/config.N_x
  dy = (config.y_start - config.y_end)/config.N_y
  N_y_local, N_x_local = da.getSizes()

  pde = Poisson2D(da, config)
  phi = da.createGlobalVec()
  rho = da.createGlobalVec()

  A = PETSc.Mat().createPython([phi.getSizes(), rho.getSizes()], comm = da.comm)
  A.setPythonContext(pde)
  A.setUp()

  ksp = PETSc.KSP().create()

  ksp.setOperators(A)
  ksp.setType('cg')

  pc = ksp.getPC()
  pc.setType('none')

  pde.formRHS(rho, rho_array)
  ksp.setFromOptions()
  ksp.solve(rho, phi)

  # Since rho was defined at (i + 0.5, j + 0.5) 
  # Electric Potential returned will also be at (i + 0.5, j + 0.5)
  electric_potential = af.to_array(np.swapaxes(phi[:].reshape(N_x_local, N_y_local), 0, 1))
  
  # Interpolating to obtain the values at (i, j):
  electric_potential = 0.25 * (electric_potential + \
                               af.shift(electric_potential, 1, 0) + \
                               af.shift(electric_potential, 0, 1) + \
                               af.shift(electric_potential, 1, 1)
                              )

  E_x = -(af.shift(electric_potential, 0, -1) - electric_potential)/dx #(i+1/2, j)
  E_y = -(af.shift(electric_potential, -1, 0) - electric_potential)/dy #(i, j+1/2)

  af.eval(E_x, E_y)
  return(E_x, E_y)

def fields_step(da, args, dt):

  config  = args.config
  N_ghost = config.N_ghost

  vel_x  = args.vel_x
  vel_y  = args.vel_y
  vel_z  = args.vel_z

  charge_electron = config.charge_electron

  # Creating local and global vectors for each of the partitioned zones:
  glob  = da.createGlobalVec()
  local = da.createLocalVec()

  # The following fields are defined on the Yee-Grid:
  E_x = args.E_x #(i + 1/2, j)
  E_y = args.E_y #(i, j + 1/2)
  E_z = args.E_z #(i, j)

  B_x = args.B_x #(i, j + 1/2)
  B_y = args.B_y #(i + 1/2, j)
  B_z = args.B_z #(i + 1/2, j + 1/2)

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  # Convert to velocitiesExpanded:
  args.f = af.moddims(args.f,                   
                      (N_x_local + 2 * N_ghost)*\
                      (N_y_local + 2 * N_ghost),\
                      config.N_vel_y,\
                      config.N_vel_x,\
                      config.N_vel_z
                     )

  from cks.fdtd import fdtd, fdtd_grid_to_ck_grid

  if(config.fields_solver == 'electrostatic'):
    
    rho_array = charge_electron * (calculate_density(args) - config.rho_background)
    #(i + 1/2, j + 1/2)
    
    rho_array = af.moddims(rho_array,\
                           N_y_local + 2 * N_ghost,\
                           N_x_local + 2 * N_ghost
                          )

    rho_array = np.array(rho_array)[N_ghost:-N_ghost,\
                                    N_ghost:-N_ghost
                                   ]
    
    E_x_local, E_y_local = solve_electrostatic_fields(da, config, rho_array)

    E_x[N_ghost:-N_ghost, N_ghost:-N_ghost] = E_x_local
    E_y[N_ghost:-N_ghost, N_ghost:-N_ghost] = E_y_local

    E_x = communicate_fields(da, config, E_x, local, glob) #(i + 1/2, j)
    E_y = communicate_fields(da, config, E_y, local, glob) #(i, j + 1/2)

  else:

    J_x = charge_electron * calculate_mom_bulk_x(args) #(i + 1/2, j + 1/2)
    J_y = charge_electron * calculate_mom_bulk_y(args) #(i + 1/2, j + 1/2)
    J_z = charge_electron * calculate_mom_bulk_z(args) #(i + 1/2, j + 1/2)

    J_x = af.moddims(J_x,\
                     N_y_local + 2 * N_ghost,\
                     N_x_local + 2 * N_ghost
                    )
    
    J_y = af.moddims(J_y,\
                     N_y_local + 2 * N_ghost,\
                     N_x_local + 2 * N_ghost
                    )

    J_z = af.moddims(J_z,\
                     N_y_local + 2 * N_ghost,\
                     N_x_local + 2 * N_ghost
                    )

    J_x = 0.5 * (J_x + af.shift(J_x, 1, 0)) #(i + 1/2, j)
    J_y = 0.5 * (J_y + af.shift(J_y, 0, 1)) #(i, j + 1/2)
    J_z = 0.25 * (J_z + af.shift(J_z, 1, 0) + af.shift(J_z, 0, 1) + af.shift(J_z, 1, 1)) #(i, j)

    J_x = communicate_fields(da, config, J_x, local, glob) #(i + 1/2, j)
    J_y = communicate_fields(da, config, J_y, local, glob) #(i, j + 1/2)
    J_z = communicate_fields(da, config, J_z, local, glob) #(i, j)

    E_x, E_y, E_z, B_x_new, B_y_new, B_z_new = fdtd(da, config,\
                                                    E_x, E_y, E_z,\
                                                    B_x, B_y, B_z,\
                                                    J_x, J_y, J_z,\
                                                    dt
                                                   )
  
    args.B_x = B_x_new #(i, j + 1/2)
    args.B_y = B_y_new #(i + 1/2, j)
    args.B_z = B_z_new #(i + 1/2, j + 1/2)

    args.E_x = E_x #(i + 1/2, j)
    args.E_y = E_y #(i, j + 1/2)
    args.E_z = E_z #(i, j)

    # To account for half-time steps:
    B_x = 0.5 * (B_x + B_x_new)
    B_y = 0.5 * (B_y + B_y_new)
    B_z = 0.5 * (B_z + B_z_new)

  E_x, E_y, E_z, B_x, B_y, B_z = fdtd_grid_to_ck_grid(da, config, E_x, E_y, E_z, B_x, B_y, B_z)

  # Tiling such that E_x, E_y and B_z have the same array dimensions as f:
  # This is required to perform the interpolation in velocity space:
  if(config.mode == '3V'):
    E_x = af.tile(af.flat(E_x), 1, vel_x.shape[1], vel_x.shape[2], vel_x.shape[3]) #(i + 1/2, j + 1/2)
    E_y = af.tile(af.flat(E_y), 1, vel_x.shape[1], vel_x.shape[2], vel_x.shape[3]) #(i + 1/2, j + 1/2)
    E_z = af.tile(af.flat(B_z), 1, vel_x.shape[1], vel_x.shape[2], vel_x.shape[3]) #(i + 1/2, j + 1/2)

    B_x = af.tile(af.flat(B_x), 1, vel_x.shape[1], vel_x.shape[2], vel_x.shape[3]) #(i + 1/2, j + 1/2)
    B_y = af.tile(af.flat(B_y), 1, vel_x.shape[1], vel_x.shape[2], vel_x.shape[3]) #(i + 1/2, j + 1/2)
    B_z = af.tile(af.flat(B_z), 1, vel_x.shape[1], vel_x.shape[2], vel_x.shape[3]) #(i + 1/2, j + 1/2)
 
  else:
    E_x = af.tile(af.flat(E_x), 1, vel_x.shape[1], vel_x.shape[2], 1) #(i + 1/2, j + 1/2)
    E_y = af.tile(af.flat(E_y), 1, vel_x.shape[1], vel_x.shape[2], 1) #(i + 1/2, j + 1/2)
    E_z = af.tile(af.flat(B_z), 1, vel_x.shape[1], vel_x.shape[2], 1) #(i + 1/2, j + 1/2)

    B_x = af.tile(af.flat(B_x), 1, vel_x.shape[1], vel_x.shape[2], 1) #(i + 1/2, j + 1/2)
    B_y = af.tile(af.flat(B_y), 1, vel_x.shape[1], vel_x.shape[2], 1) #(i + 1/2, j + 1/2)
    B_z = af.tile(af.flat(B_z), 1, vel_x.shape[1], vel_x.shape[2], 1) #(i + 1/2, j + 1/2)

  F_x = charge_electron * (E_x) #+ vel_y * B_z - vel_z * B_y) #(i + 1/2, j + 1/2)
  F_y = charge_electron * (E_y) #- vel_x * B_z + vel_z * B_x) #(i + 1/2, j + 1/2)
  F_z = charge_electron * (E_z) #- vel_y * B_x + vel_x * B_y) #(i + 1/2, j + 1/2)

  args.f = f_interp_vel_3d(args, F_x, F_y, F_z, dt)

  # Convert to positionsExpanded:
  args.f = af.moddims(args.f,                   
                      (N_y_local + 2 * config.N_ghost),\
                      (N_x_local + 2 * config.N_ghost),\
                      config.N_vel_y*config.N_vel_x*config.N_vel_z,\
                      1
                     )
  af.eval(args.f)

  # Destroying the vectors since we are done with their usage for the time-step:
  glob.destroy()
  local.destroy()

  return(args)

def time_integration(da, da_fields, args, time_array):

  data = np.zeros(time_array.size)

  glob  = da.createGlobalVec()
  local = da.createLocalVec()

  N_ghost = args.config.N_ghost

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  N_vel_x = args.config.N_vel_x
  N_vel_y = args.config.N_vel_y
  N_vel_z = args.config.N_vel_z

  # Convert to velocitiesExpanded:
  args.f = af.moddims(args.f,                   
                      (N_x_local + 2 * N_ghost)*\
                      (N_y_local + 2 * N_ghost),\
                      N_vel_y,\
                      N_vel_x,\
                      N_vel_z
                     )

  # Storing the value of density amplitude at t = 0
  data[0] = af.max(calculate_density(args))

  # Convert to positionsExpanded:
  args.f = af.moddims(args.f,                   
                      (N_y_local + 2 * N_ghost),\
                      (N_x_local + 2 * N_ghost),\
                      N_vel_y*N_vel_x*N_vel_z,\
                      1
                     )

  from cks.interpolation_routines import f_interp_2d
  
  for time_index, t0 in enumerate(time_array[1:]):
    # Printing progress every 10 iterations
    # Printing only at rank = 0 to avoid multiple outputs:
    
    if(time_index%10 == 0 and da.getComm().rank == 0):
        print("Computing for Time =", t0)

    dt = time_array[1] - time_array[0]

    args.f = f_interp_2d(da, args, 0.25*dt)
    args.f = communicate_distribution_function(da, args, local, glob)
    args.f = collision_step(da, args, 0.5*dt)
    args.f = communicate_distribution_function(da, args, local, glob)
    args.f = f_interp_2d(da, args, 0.25*dt)
    args.f = communicate_distribution_function(da, args, local, glob)
    args   = fields_step(da_fields, args, dt)
    args.f = communicate_distribution_function(da, args, local, glob)
    args.f = f_interp_2d(da, args, 0.25*dt)
    args.f = communicate_distribution_function(da, args, local, glob)
    args.f = collision_step(da, args, 0.5*dt)
    args.f = communicate_distribution_function(da, args, local, glob)
    args.f = f_interp_2d(da, args, 0.25*dt)
    args.f = communicate_distribution_function(da, args, local, glob)
    
    # Convert to velocitiesExpanded:
    args.f = af.moddims(args.f,                   
                        (N_x_local + 2 * N_ghost)*\
                        (N_y_local + 2 * N_ghost),\
                        N_vel_y,\
                        N_vel_x,\
                        N_vel_z
                       )

    data[time_index + 1] = af.max(calculate_density(args))

    # Convert to positionsExpanded:
    args.f = af.moddims(args.f,                   
                        (N_y_local + 2 * N_ghost),\
                        (N_x_local + 2 * N_ghost),\
                        N_vel_y*N_vel_x*N_vel_z,\
                        1
                       )

  glob.destroy()
  local.destroy()

  return(data, args.f)