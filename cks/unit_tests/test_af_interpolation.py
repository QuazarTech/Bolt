import numpy as np
import arrayfire as af

def test_1d():
  N_x      = np.arange(500, 2000, 20)
  x_interp = af.randu(10, dtype=af.Dtype.f64)
  error    = np.zeros(N_x.size)

  for i in range(N_x.size):
    x  = af.to_array(np.linspace(0, 1, N_x[i]))
    dx = af.sum(x[1])

    y  = af.sin(2*np.pi*x)

    y_interp = af.approx1(y, x_interp/dx , af.INTERP.CUBIC_SPLINE)
    y_ana    = af.sin(2*np.pi*x_interp)

    error[i] = af.sum(af.abs(y_interp - y_ana))/10

  x    = np.log10(N_x)
  poly = np.polyfit(x, np.log10(error), 1)

  assert(abs(poly[0] + 3)<0.1)

def test_2d():
  N_x  = np.arange(500, 2000, 20)
  N_y  = np.arange(500, 2000, 20)

  x_interp = af.randu(10, dtype=af.Dtype.f64)
  y_interp = af.randu(10, dtype=af.Dtype.f64)

  error    = np.zeros(N_x.size)

  for i in range(N_x.size):
    x  = af.to_array(np.linspace(0, 1, N_x[i]))
    y  = af.to_array(np.linspace(0, 2, N_x[i]))
    
    dx = af.sum(x[1])
    dy = af.sum(y[1])

    x  = af.tile(x, 1, int(N_y[i]))
    y  = af.tile(af.reorder(y), int(N_x[i]), 1)
    
    z  = af.sin(2*np.pi*x + 4*np.pi*y)

    z_interp = af.approx2(z, x_interp/dx , y_interp/dy, af.INTERP.BICUBIC_SPLINE)
    z_ana    = af.sin(2*np.pi*x_interp + 4*np.pi*y_interp)

    error[i] = af.sum(af.abs(z_interp - z_ana))/10

  x    = np.log10(N_x)
  poly = np.polyfit(x, np.log10(error), 1)

  assert(abs(poly[0] + 3)<0.1)