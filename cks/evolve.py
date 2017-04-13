import numpy as np
import arrayfire as af
import cks.initialize as initialize
from scipy.fftpack import fftfreq

def calculate_density(args):
  """
  This function evaluates and returns the density in the 1D/2D space, depending on the
  dimensionality of the system studied.

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    f : 2D/4D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    vel_x : 2D/4D velocity array which has the variations in x-velocity along axis 1 in the
            2D case with the variations along axis 2 in the 4D case.

    vel_y : If applicable it is the 4D velocity array which has the variations in y-velocity 
            along axis 3
  
  Output:
  -------
    density : The density array is returned by this function after computing the moments. The 
              values in the array indicate the values of density with changes in x and y.
  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x

  if(config.mode == '2D2V'):
    vel_y = args.vel_y
    dv_x  = af.sum(vel_x[0, 0, 1, 0]-vel_x[0, 0, 0, 0])
    dv_y  = af.sum(vel_y[0, 0, 0, 1]-vel_y[0, 0, 0, 0])

    density = af.sum(af.sum(f, 2)*dv_x, 3)*dv_y
  
  else:
    dv_x    = af.sum(vel_x[0, 1]-vel_x[0, 0])
    density = af.sum(f, 1)*dv_x

  af.eval(density)
  return(density)

def calculate_vel_bulk_x(args):
  """
  This function evaluates and returns the x-component of bulk velocity in the 1D/2D space, 
  depending on the dimensionality of the system studied.

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    f : 2D/4D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    vel_x : 2D/4D velocity array which has the variations in x-velocity along axis 1 in the
            2D case with the variations along axis 2 in the 4D case.

    vel_y : If applicable: it is the 4D velocity array which has the variations in y-velocity 
            along axis 3
  
  Output:
  -------
    vel_bulk_x : The vel_bulk_x array is returned by this function after computing the moments. The 
                 values in the array indicate the values of vel_bulk_x with changes in x and y.
  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x


  if(config.mode == '2D2V'):
    vel_y = args.vel_y
    dv_x  = af.sum(vel_x[0, 0, 1, 0]-vel_x[0, 0, 0, 0])
    dv_y  = af.sum(vel_y[0, 0, 0, 1]-vel_y[0, 0, 0, 0])

    momentum_x = af.sum(af.sum(f * vel_x, 2)*dv_x, 3)*dv_y
    vel_bulk_x = momentum_x/calculate_density(args)
  
  else:
    dv_x        = af.sum(vel_x[0, 1]-vel_x[0, 0])
    momentum_x  = af.sum(f*vel_x, 1)*dv_x
    vel_bulk_x  = momentum_x/calculate_density(args)
  
  af.eval(vel_bulk_x)
  return(vel_bulk_x)

def calculate_vel_bulk_y(args):
  """
  This function evaluates and returns the y-component of bulk velocity in the 4D space.

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    f : 4D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    vel_x : 4D velocity array which has the variations in x-velocity along axis 2

    vel_y : 4D velocity array which has the variations in y-velocity along axis 3
  
  Output:
  -------
    vel_bulk_y : The vel_bulk_y array is returned by this function after computing the moments. The 
                 values in the array indicate the values in the y-component of bulk velocity with 
                 changes in x and y.
  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y

  dv_x = af.sum(vel_x[0, 0, 1, 0]-vel_x[0, 0, 0, 0])
  dv_y = af.sum(vel_y[0, 0, 0, 1]-vel_y[0, 0, 0, 0])

  momentum_y = af.sum(af.sum(f * vel_y, 2)*dv_x, 3)*dv_y
  vel_bulk_y = momentum_y/calculate_density(args)
  
  af.eval(vel_bulk_y)
  return(vel_bulk_y)

def calculate_temperature(args):
  """
  This function evaluates and returns the temperature in the 1D/2D space, depending on the
  dimensionality of the system studied.

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    f : 2D/4D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    vel_x : 2D/4D velocity array which has the variations in x-velocity along axis 1 in the
            2D case with the variations along axis 2 in the 4D case.

    vel_y : If applicable it is the 4D velocity array which has the variations in y-velocity 
            along axis 3
  
  Output:
  -------
    temperature : The temperature array is returned by this function after computing the moments. The 
                  values in the array indicate the values of temperature with changes in x and y.
  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x

  if(config.mode == '2D2V'):
    vel_y = args.vel_y
    dv_x  = af.sum(vel_x[0, 0, 1, 0]-vel_x[0, 0, 0, 0])
    dv_y  = af.sum(vel_y[0, 0, 0, 1]-vel_y[0, 0, 0, 0])

    vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, 1, f.shape[2], f.shape[3])
    vel_bulk_y = af.tile(calculate_vel_bulk_y(args), 1, 1, f.shape[2], f.shape[3])

    pressure    = 0.5 * af.sum(af.sum(f*((vel_x-vel_bulk_x)**2 + (vel_y-vel_bulk_y)**2), 2)*dv_x, 3)*dv_y
    temperature = pressure/calculate_density(args)
  
  else:
    dv_x        = af.sum(vel_x[0, 1]-vel_x[0, 0])
    vel_bulk_x  = af.tile(calculate_vel_bulk_x(args), 1, vel_x.shape[1])
    temperature = af.sum(f*(vel_x-vel_bulk_x)**2, 1)*dv_x
    temperature = temperature/calculate_density(args)

  af.eval(temperature)
  return(temperature)

def f_MB(args):
  """
  Return the local Maxwell Boltzmann distribution function. This distribution function is 
  Maxwellian while maintaining the bulk parameters of the original distribution function which
  was passed to it.
  
  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    config: Object config which is obtained by set() is passed to this file

    f : 2D/4D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    vel_x : 2D/4D velocity array which has the variations in x-velocity along axis 1/2

    vel_y : If applicable it is the 4D velocity array which has the variations in 
            y-velocity along axis 3

  Output:
  -------
    f_MB : Local Maxwell Boltzmann distribution array.
  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x

  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  if(config.mode == '2D2V'):
    vel_y      = args.vel_y
    n          = af.tile(calculate_density(args), 1, 1, f.shape[2], f.shape[3])
    T          = af.tile(calculate_temperature(args), 1, 1, f.shape[2], f.shape[3])
    vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, 1, f.shape[2], f.shape[3])
    vel_bulk_y = af.tile(calculate_vel_bulk_y(args), 1, 1, f.shape[2], f.shape[3])
    
    f_MB = n * (mass_particle/(2*np.pi*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_x - vel_bulk_x)**2/(2*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_y - vel_bulk_y)**2/(2*boltzmann_constant*T))  
  
  else:
    n          = af.tile(calculate_density(args), 1, f.shape[1])
    T          = af.tile(calculate_temperature(args), 1, f.shape[1])
    vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, f.shape[1])
    
    f_MB = n*af.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T))*\
             af.exp(-mass_particle*(vel_x-vel_bulk_x)**2/(2*boltzmann_constant*T))

  af.eval(f_MB)
  return(f_MB)

def f_interp2(args, dt):
  """
  Performs the advection step in 2D + 2V space.
  
  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    config: Object config which is obtained by set() is passed to this file

    f : 4D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    vel_x : 4D velocity array which has the variations in x-velocity along axis 2

    vel_y : 4D velocity array which has the variations in y-velocity along axis 3

    x : 4D array that contains the variations in x along the 1st axis

    y : 4D array that contains the variations in y along the 0th axis

    dt : Time step for which the system is evolved forward

  Output:
  -------
    f_interp : Returns the distribution function after performing the interpolation step
  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y
  x      = args.x 
  y      = args.y 

  left_boundary  = config.left_boundary
  right_boundary = config.right_boundary
  
  bot_boundary = config.bot_boundary
  top_boundary = config.top_boundary

  length_x = right_boundary - left_boundary
  length_y = top_boundary - bot_boundary

  N_ghost_x = config.N_ghost_x
  N_ghost_y = config.N_ghost_y

  N_x = config.N_x
  N_y = config.N_y

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y

  f_interp  = af.constant(0, N_y + 2*N_ghost_y, N_x + 2*N_ghost_x, N_vel_x, N_vel_y, dtype = af.Dtype.f64)
  
  x_new  = x - vel_x*dt
  dx     = af.sum(x[0, 1, 0, 0] - x[0, 0, 0, 0])  
 
  while(af.sum(x_new<left_boundary)!=0):
      x_new = af.select(x_new<left_boundary,
                        x_new + length_x,
                        x_new
                       )

  while(af.sum(x_new>right_boundary)!=0):
      x_new = af.select(x_new>right_boundary,
                        x_new - length_x,
                        x_new
                       )

  y_new  = y - vel_y*dt
  dy     = af.sum(y[1, 0, 0, 0] - y[0, 0, 0, 0])  

  while(af.sum(y_new<bot_boundary)!=0):
      y_new = af.select(y_new<bot_boundary,
                        y_new + length_y,
                        y_new
                       )

  while(af.sum(y_new>top_boundary)!=0):
      y_new = af.select(y_new>top_boundary,
                        y_new - length_y,
                        y_new
                       )

  x_interpolant = x_new/dx
  y_interpolant = y_new/dy
  
  f_interp = af.approx2(f,\
                        y_interpolant,\
                        x_interpolant,\
                        af.INTERP.BICUBIC_SPLINE
                       )
  
  af.eval(f_interp)
  return(f_interp)

def f_interp(args, dt):
  """
  Performs the advection step in 1D + 1V space.
  
  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    config: Object config which is obtained by set() is passed to this file

    f : 2D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    vel_x : 2D velocity array which has the variations in x-velocity along axis 1

    x : 2D array that contains the variations in x along the 0th axis

    dt : Time step for which the system is evolved forward

  Output:
  -------
    f_interp : Returns the distribution function after performing the interpolation step
  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x
  x      = args.x 

  left_boundary  = config.left_boundary
  right_boundary = config.right_boundary
  length_x       = right_boundary - left_boundary

  N_ghost_x = config.N_ghost_x

  x_new     = x - vel_x*dt
  step_size = af.sum(x[1,0] - x[0,0])
  f_interp  = af.constant(0, f.shape[0], f.shape[1], dtype = af.Dtype.f64)
  
  # Interpolating:
  
  x_temp = x_new[N_ghost_x:-N_ghost_x, :]
  
  while(af.sum(x_temp<left_boundary)!=0):
      x_temp = af.select(x_temp<left_boundary,
                         x_temp + length_x,
                         x_temp
                        )

  while(af.sum(x_temp>right_boundary)!=0):
      x_temp = af.select(x_temp>right_boundary,
                         x_temp - length_x,
                         x_temp
                        )
  
  x_interpolant = x_temp/step_size + N_ghost_x
  
  f_interp[N_ghost_x:-N_ghost_x, :] = af.approx1(f, x_interpolant,\
                                                 af.INTERP.CUBIC_SPLINE
                                                )
 
  af.eval(f_interp)
  
  return(f_interp)

def collision_step(args, dt):
  """
  Performs the collision step where df/dt = C[f] is solved for
  In this function the BGK collision operator is being solved for.

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    config: Object config which is obtained by set() is passed to this file

    f : 2D/4D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    dt : Time step for which the system is evolved forward

  Output:
  -------
    f_final : Returns the distribution function after performing the collision step
  """
  tau = args.config.tau
  f   = args.f 

  f0             = f_MB(args)
  f_intermediate = f - (dt/2)*(f - f0)/tau
  f_final        = f - (dt)*(f_intermediate - f0)/tau

  af.eval(f_final)
  return(f_final)

def fft_poisson(rho, dx, dy = None):
  """
  FFT solver which returns the value of electric field. This will only work
  when the system being solved for has periodic boundary conditions.

  Parameters:
  -----------
    rho : The 1D/2D density array obtained from calculate_density() is passed to this
          function.

    dx  : Step size in the x-grid

    dy  : Step size in the y-grid.Set to None by default to avoid conflict with the 1D case.

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
    
    potential_hat       = (1/(4 * np.pi**2 * (k_x**2 + k_y**2))) * rho_hat
    potential_hat[0, :] = 0
    potential_hat[:, 0] = 0
    
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


def f_interp2_v(args, E_x, E_y, dt):
  """
  Performs the interpolation in 2V velocity space. This function is
  used in solving for the fields contribution in the Boltzmann equation.

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    config: Object config which is obtained by set() is passed to this file

    f : 4D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    vel_x : 4D velocity array which has the variations in x-velocity along axis 2

    vel_y : 4D velocity array which has the variations in y-velocity along axis 3

    E_x : x-component of the electric field. 

    E_y : y-component of the electric field.
    
    dt : Time step for which the system is evolved forward

  Output:
  -------
    f_interp : Returns the distribution function after performing the interpolation step
               in velocity space.

  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y
  
  charge_particle = config.charge_particle
  vel_x_max       = config.vel_x_max
  vel_y_max       = config.vel_y_max

  vel_x_new = vel_x - charge_particle * dt * af.tile(E_x, 1, 1, f.shape[2], f.shape[3])
  vel_y_new = vel_y - charge_particle * dt * af.tile(E_y, 1, 1, f.shape[2], f.shape[3])

  dv_x = af.sum(vel_x[0, 0, 1, 0]-vel_x[0, 0, 0, 0])
  dv_y = af.sum(vel_y[0, 0, 0, 1]-vel_y[0, 0, 0, 0])
     
  vel_x_interpolant = (vel_x_new + vel_x_max)/dv_x
  vel_y_interpolant = (vel_y_new + vel_y_max)/dv_y
  
  f_interp      = af.approx2(af.reorder(f, 2, 3, 0, 1),\
                             af.reorder(vel_x_interpolant, 2, 3, 0, 1),\
                             af.reorder(vel_y_interpolant, 2, 3, 0, 1),\
                             af.INTERP.BICUBIC_SPLINE
                            )
  
  f_interp = af.reorder(f_interp, 2, 3, 0, 1)

  af.eval(f_interp)
  return(f_interp)

def f_interp_v(args, E_x, dt):
  """
  Performs the interpolation in 1V velocity space. This function is
  used in solving for the fields contribution in the Boltzmann equation.

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    config: Object config which is obtained by set() is passed to this file

    f : 2D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    vel_x : 2D velocity array which has the variations in x-velocity along axis 1

    E_x : x-component of the electric field. 

    dt : Time step for which the system is evolved forward

  Output:
  -------
    f_interp : Returns the distribution function after performing the interpolation step
               in velocity space.
  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x

  charge_particle = config.charge_particle
  vel_x_max       = config.vel_x_max

  v_new     = vel_x - charge_particle * dt * af.tile(E_x, 1, f.shape[1])
  step_size = af.sum(vel_x[0,1] - vel_x[0,0])
  
  # Interpolating:
     
  v_interpolant = (v_new + vel_x_max)/step_size
  
  f_interp      = af.approx1(af.reorder(f),\
                             af.reorder(v_interpolant),\
                             af.INTERP.CUBIC_SPLINE
                            )
  
  f_interp = af.reorder(f_interp)

  af.eval(f_interp)

  return f_interp

def periodic_x(config, f):
  """
  Applied to periodic boundary conditions in x to the function that
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
    f[:N_ghost_x,:]   = f[-(2*N_ghost_x + 1):-(N_ghost_x + 1)]
    f[-N_ghost_x:, :] = f[(N_ghost_x + 1):(2*N_ghost_x + 1)]  

  af.eval(f)
  return(f)

def periodic_y(config, f):
  """
  Applied to periodic boundary conditions in y to the function that
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

def fields_step(args, dt):
  """
  Solves for the field step where df/dt + E df/dv is solved for

  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    config: Object config which is obtained by set() is passed to this file

    f : 2D/4D distribution function that is passed to the function. Moments will be computed
        defined by the state of the system which is indicated by the distribution function

    vel_x : 2D/4D velocity array which has the variations in x-velocity along axis 1/2

    vel_y : If applicable it is the 4D velocity array which has the variations in 
            y-velocity along axis 3

    dt : Time step for which the system is evolved forward

  Output:
  -------
    f_fields : Returns the distribution function after performing the fields step.
  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x
  
  N_x       = config.N_x
  N_ghost_x = config.N_ghost_x

  left_boundary  = config.left_boundary
  right_boundary = config.right_boundary
  length_x       = right_boundary - left_boundary
  dx             = length_x/(N_x - 1) 
  
  charge_particle = config.charge_particle
  
  if(config.mode == '2D2V'):
    vel_y     = args.vel_y
    N_y       = config.N_y
    N_ghost_y = config.N_ghost_y

    bot_boundary = config.bot_boundary
    top_boundary = config.top_boundary
    length_y     = top_boundary - bot_boundary
    dy           = length_y/(N_y - 1) 

    E_x = af.constant(0, f.shape[0], f.shape[1], dtype = af.Dtype.c64)
    E_y = af.constant(0, f.shape[0], f.shape[1], dtype = af.Dtype.c64)
    
    E_x_local, E_y_local = fft_poisson(charge_particle*\
                                       calculate_density(args)[N_ghost_y:-N_ghost_y-1, N_ghost_x:-N_ghost_x - 1],\
                                       dx,\
                                       dy
                                      )
    
    E_x_local = af.join(0, E_x_local, E_x_local[0])
    E_x_local = af.join(1, E_x_local, E_x_local[:, 0])
    
    E_y_local = af.join(0, E_y_local, E_y_local[0])
    E_y_local = af.join(1, E_y_local, E_y_local[:, 0])

    E_x[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] = E_x_local
    E_x                                             = periodic_x(config, E_x)
    E_x                                             = periodic_y(config, E_x)

    E_y[N_ghost_y:-N_ghost_y, N_ghost_x:-N_ghost_x] = E_y_local
    E_y                                             = periodic_x(config, E_y)
    E_y                                             = periodic_y(config, E_y)
    
    f_fields = f_interp2_v(args, af.real(E_x), af.real(E_y), dt)

  else:
    E_x       = af.constant(0, f.shape[0], dtype = af.Dtype.c64)
    E_x_local = fft_poisson(charge_particle*
                            calculate_density(args)[N_ghost_x:-N_ghost_x-1],\
                            dx
                           )
    E_x_local = af.join(0, E_x_local, E_x_local[0])
    
    E_x[N_ghost_x:-N_ghost_x] = E_x_local
    E_x                       = periodic_x(config, E_x)     
    
    f_fields = f_interp_v(args, af.real(E_x), dt)
    
  af.eval(f_fields)
  return f_fields

def time_integration(args, time_array):
  """
  The main function that evolves the system in time.
  
  Parameters:
  -----------
    Object args is passed to the function of which the following attributes are utilized:

    config: Object config which is obtained by set() is passed to this file

    f_initial : 2D/4D distribution function that is passed to the function. This is the 
                initial condition for the simulation, and is typically obtained by using 
                the appropriately named function from the initialize sub-module.

    vel_x : 2D/4D velocity array which has the variations in x-velocity along axis 1/2

    vel_y : If applicable it is the 4D velocity array which has the variations in 
            y-velocity along axis 3

    x : 2D/4D array that contains the variations in x along the 0th/1st axis

    y : If applicable it is the 4D array that contains the variations in y 
        along the 0th axis
    
    time_array : Array that contains the values of time at which the 
                 simulation evaluates the physical quantities.  

  Output:
  -------
    data : Contains data about the amplitude of density in the system with progression
           in time.

    f_current : This array returned is the distribution function that is obtained
                in the final time step.
  """
  config = args.config
  f      = args.f
  vel_x  = args.vel_x
  x      = args.x 
  
  if(config.mode == '2D2V'):
    y      = args.y 
    vel_y  = args.vel_y
  
  data      = np.zeros(time_array.size)

  fields_enabled     = config.fields_enabled
  collisions_enabled = config.collisions_enabled

  for time_index, t0 in enumerate(time_array):
    if(time_index%10 == 0):
        print("Computing for Time = ", t0)
    # We shall split the Boltzmann-Equation and solve it:
    # In this step we are solving the collisionless equation
    dt = time_array[1] - time_array[0]
    
    if(config.mode == '2D2V'):
      args.f = f_interp2(args, 0.25*dt)
    else:
      args.f = f_interp(args, 0.25*dt)
    
    args.f = periodic_x(config, args.f)

    if(config.mode == '2D2V'):
      args.f = periodic_y(config, args.f)

    if(collisions_enabled == "True"):
      args.f = collision_step(args, 0.5*dt)

    args.f = periodic_x(config, args.f)
    
    if(config.mode == '2D2V'):
      args.f = periodic_y(config, args.f)

    if(config.mode == '2D2V'):
      args.f = f_interp2(args, 0.25*dt)
    else:
      args.f = f_interp(args, 0.25*dt)
    
    args.f = periodic_x(config, args.f)

    if(config.mode == '2D2V'):
      args.f = periodic_y(config, args.f)
    
    if(collisions_enabled == "True"):
      args.f = collision_step(args, 0.5*dt)
    
    if(fields_enabled == "True"):
      args.f = fields_step(args, dt)

    args.f = periodic_x(config, args.f)
    
    if(config.mode == '2D2V'):
      args.f = periodic_y(config, args.f)

    if(config.mode == '2D2V'):
      args.f = f_interp2(args, 0.25*dt)
    else:
      args.f = f_interp(args, 0.25*dt)
    
    args.f = periodic_x(config, args.f)

    if(config.mode == '2D2V'):
      args.f = periodic_y(config, args.f)

    if(collisions_enabled == "True"):
      args.f = collision_step(args, 0.5*dt)
    
    args.f = periodic_x(config, args.f)

    if(config.mode == '2D2V'):
      args.f = periodic_y(config, args.f)

    if(config.mode == '2D2V'):
      args.f = f_interp2(args, 0.25*dt)
    else:
      args.f = f_interp(args, 0.25*dt)
    
    args.f = periodic_x(config, args.f)

    if(config.mode == '2D2V'):
      args.f = periodic_y(config, args.f)

    data[time_index] = af.max(calculate_density(args))

  return(data, args.f)