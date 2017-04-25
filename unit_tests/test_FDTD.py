import numpy as np
import arrayfire as af
import time as timer
import copy
af.set_backend("cpu")

import cks.initialize as initialize
import cks.fdtd as fdtd
import cks.evolve as evolve
import params

def gauss1D(x, spread):
  return af.exp(-((x - 0.5)**2 )/(2*spread**2))

testlist   = [initialize.config() for count in range(4)]

testlist[0].N_x = testlist[0].N_y = 32
testlist[1].N_x = testlist[1].N_y = 64
testlist[2].N_x = testlist[2].N_y = 128
testlist[3].N_x = testlist[3].N_y = 256
# testlist[4].N_x = testlist[4].N_y = 512

for item in testlist:
  item.N_ghost_x      = item.N_ghost_y    = 1
  item.left_boundary  = item.bot_boundary = 0
  item.right_boundary = item.top_boundary = 1
  item.mode           = '2D2V'

error_E_x = np.zeros(len(testlist))
error_E_y = np.zeros(len(testlist))
error_B_z = np.zeros(len(testlist))

i     = 0

for config in testlist:
  N_x = config.N_x
  N_y = config.N_y

  dx = 1/(N_x - 1)
  dy = 1/(N_y - 1)

  x_center = np.linspace(-dx, 1 + dx, N_x + 2)
  y_center = np.linspace(-dy, 1 + dy, N_y + 2)

  x_right = np.linspace(-0.5*dx, 1 + 1.5*dx, N_x + 2)
  y_top   = np.linspace(-0.5*dy, 1 + 1.5*dy, N_y + 2)

  x_center = af.to_array(x_center)
  y_center = af.to_array(y_center)
  x_right  = af.to_array(x_right)
  y_top    = af.to_array(y_top)

  print(N_x, N_y)

  final_time = 1
  dt         = (dx/2)
  time       = np.arange(0, final_time, dt)

  Ez = af.data.constant(0,x_center.elements(),y_center.elements(), dtype=af.Dtype.f64)
  Bx = af.data.constant(0,x_center.elements(),y_center.elements(), dtype=af.Dtype.f64)
  By = af.data.constant(0,x_center.elements(),y_center.elements(), dtype=af.Dtype.f64)

  Bz = af.data.constant(0,x_center.elements(),y_center.elements(), dtype=af.Dtype.f64)
  Ex = af.data.constant(0,x_center.elements(),y_center.elements(), dtype=af.Dtype.f64)
  Ey = af.data.constant(0,x_center.elements(),y_center.elements(), dtype=af.Dtype.f64)

  X_center = af.tile(af.reorder(x_center),y_center.elements(),1)
  X_right  = af.tile(af.reorder(x_right), y_center.elements(),1)
  
  Y_center = af.tile(y_center, 1, x_center.elements())
  Y_top    = af.tile(y_top,    1, x_center.elements())

  Ex[1:-1, 1:-1] = gauss1D(Y_center[1:-1, 1:-1], 0.1)
  Ey[1:-1, 1:-1] = gauss1D(X_center[1:-1, 1:-1], 0.1)

  Ex_initial = Ex.copy()
  Ey_initial = Ey.copy()
  Ez_initial = Ez.copy()
  Bx_initial = Bx.copy()
  By_initial = By.copy()
  Bz_initial = Bz.copy()

  Jx, Jy, Jz = 0, 0, 0

  for time_index, t0 in enumerate(time):
    Bz, Ex, Ey = fdtd.mode2_fdtd(config, Bz, Ex, Ey, 0, 0, dt)

  error_E_x[i] = af.sum(af.abs(Ex[1:-1, 1:-1] - Ex_initial[1:-1, 1:-1]))/(X_center.shape[0] * X_center.shape[1])
  error_E_y[i] = af.sum(af.abs(Ey[1:-1, 1:-1] - Ey_initial[1:-1, 1:-1]))/(X_center.shape[0] * X_center.shape[1])
  error_B_z[i] = af.sum(af.abs(Bz[1:-1, 1:-1] - Bz_initial[1:-1, 1:-1]))/(X_center.shape[0] * X_center.shape[1])
  i = i + 1
  
print(error_E_x)
print(error_E_y)
print(error_B_z)

import pylab as pl 
N = np.array([32, 64, 128, 256])
pl.loglog(N, error_E_x)
pl.loglog(N, error_E_y)
pl.loglog(N, error_B_z)
pl.loglog(N, 1/N**2, '--', color = 'black')
pl.show()