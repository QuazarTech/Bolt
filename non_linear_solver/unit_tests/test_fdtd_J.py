import numpy as np
import arrayfire as af
import cks.initialize as initialize
import cks.fdtd as fdtd
import params_files.N_512 as N_512
import pylab as pl 
from cks.poisson_solvers import fft_poisson
from cks.boundary_conditions.periodic import periodic_x, periodic_y

pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20  
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8     
pl.rcParams['xtick.minor.size'] = 4     
pl.rcParams['xtick.major.pad']  = 8     
pl.rcParams['xtick.minor.pad']  = 8     
pl.rcParams['xtick.color']      = 'k'     
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'    

pl.rcParams['ytick.major.size'] = 8     
pl.rcParams['ytick.minor.size'] = 4     
pl.rcParams['ytick.major.pad']  = 8     
pl.rcParams['ytick.minor.pad']  = 8     
pl.rcParams['ytick.color']      = 'k'     
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in' 

config = initialize.set(N_512)

X_left   = initialize.calculate_x(config)[:, :, 0, 0]
Y_bottom = initialize.calculate_y(config)[:, :, 0, 0]

dx = af.sum(X_left[0, 1, 0, 0]   - X_left[0, 0, 0, 0])
dy = af.sum(Y_bottom[1, 0, 0, 0] - Y_bottom[0, 0, 0, 0])

X_center = X_left   + (dx/2)
Y_center = Y_bottom + (dy/2)

def rho(t, X, Y):
  # rho = 0.03 * np.cos(2*np.pi*t) * af.cos(2*np.pi*X + 4*np.pi*Y)
  rho = 0*2*np.pi*(af.cos(2*np.pi*X) - af.sin(2*np.pi*Y))
  return(rho)

def J_x(t, X, Y):
  J_x = 0.01 * np.sin(2*np.pi*t) * af.sin(2*np.pi*X + 4*np.pi*Y)
  return(J_x)

def J_y(t, X, Y):
  J_y = 0.01 * np.sin(2*np.pi*t) * af.sin(2*np.pi*X + 4*np.pi*Y)
  return(J_y)

def E_x(X, Y):
  # E_x = 0.03*af.sin(2*np.pi*X + 4*np.pi*Y) * 2*np.pi/(4*np.pi**2 + 16*np.pi**2)
  E_x = af.sin(2*np.pi*Y)
  return(E_x)

def E_y(X, Y):
  # E_y = 0.03*af.sin(2*np.pi*X + 4*np.pi*Y) * 4*np.pi/(4*np.pi**2 + 16*np.pi**2)
  E_y = af.cos(2*np.pi*X)
  return(E_y)


# E_x_local, E_y_local = fft_poisson(rho(0, x, y), dx, dy)

# E_y_local = af.join(0, E_y_local, E_y_local[0])
# E_y_local = af.join(1, E_y_local, E_y_local[:, 0])

# E_y = af.constant(0, 1030, 1030, dtype=af.Dtype.c64)

# E_y[3:-3, 3:-3] = E_y_local

# E_x_local = af.join(0, E_x_local, E_x_local[0])
# E_x_local = af.join(1, E_x_local, E_x_local[:, 0])

# E_x = af.constant(0, 1030, 1030, dtype=af.Dtype.c64)

# E_x[3:-3, 3:-3] = E_x_local

# E_x = periodic_x(config, E_x)
# E_x = periodic_y(config, E_x)

# E_y = periodic_x(config, E_y)
# E_y = periodic_y(config, E_y)

Ex_center_bottom = E_x(X_center, Y_bottom)
Ey_left_center   = E_y(X_left,   Y_center)

# Ex = 0.5 * (Ex + af.shift(Ex, 0, -1))
# Ex = periodic_x(config, Ex)
# Ex = periodic_y(config, Ex)

# Ey = 0.5 * (Ey + af.shift(Ey, -1, 0))
# Ey = periodic_x(config, Ey)
# Ey = periodic_y(config, Ey)

Bz = af.constant(0, Ex_center_bottom.shape[0], Ex_center_bottom.shape[1])

gradE =  (Ex_center_bottom - af.shift(Ex_center_bottom, 0, 1))/(dx) \
       + (Ey_left_center   - af.shift(Ey_left_center,   1, 0))/(dy)

# pl.contourf(np.array(gradE)[3:-3, 3:-3], 100)
# pl.colorbar()
# pl.show()

# pl.contourf(np.array(rho(0, X, Y))[3:-3, 3:-3], 100)
# pl.colorbar()
# pl.show()

pl.contourf(np.array(X_left[3:-3, 3:-3]), np.array(Y_bottom[3:-3, 3:-3]),\
            (abs(np.array(gradE) - np.array(rho(0, X_left, Y_bottom))))[3:-3, 3:-3],\
            100
           )

pl.ylabel('$y$')
pl.xlabel('$x$')

pl.title(r'$\log(|\nabla\cdot E - \rho|)$')
pl.colorbar()
pl.savefig('plot.png')

dt      = 0.001
t_final = 0.01
time    = np.arange(dt, t_final + dt, dt)

Ex = Ex_center_bottom
Ey = Ey_left_center

for time_index, t0 in enumerate(time):
  Bz, Ex, Ey = fdtd.mode2_fdtd(config, Bz, Ex, Ey,\
                               J_x(t0 - dt, X_center, Y_center),\
                               J_y(t0 - dt, X_left, Y_center), dt
                              )

gradE = af.convolve2_separable(af.Array([0,  1, 0]), af.Array([1, -1, 0]), Ex)/(dx) +\
        af.convolve2_separable(af.Array([1, -1, 0]), af.Array([0,  1, 0]), Ey)/(dy)

gradE = periodic_x(config, gradE)
gradE = periodic_y(config, gradE)

pl.contourf((abs(np.array(gradE) - np.array(rho(t0)))), 100)
pl.colorbar()
pl.show()