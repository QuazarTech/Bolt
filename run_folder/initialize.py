import arrayfire as af
import numpy as np 

def intial_conditions(q1, q2, p1, p2, p3, params):

  m = params.mass_particle
  k = params.boltzmann_constant

  rho_b = params.rho_background
  T_b   = params.temperature_background

  p1_bulk = params.p1_bulk_background
  p2_bulk = params.p2_bulk_background
  p3_bulk = params.p3_bulk_background

  pert_real = params.pert_real
  pert_imag = params.pert_imag

  k_q1 = params.k_q1
  k_q2 = params.k_q2

  # Assigning 'be'(backend) to the np/af module object in the case of the linear/nonlinear code:
  if(str(type(q1)) =="<class 'numpy.ndarray'>"):
    be = np 
  else:
    be = af

  # Calculating the perturbed density:
  rho = rho_b + (pert_real * be.cos(k_q1 * q1 + k_q2 * q2) -\
                 pert_imag * be.sin(k_q1 * q1 + k_q2 * q2)
                )

  # Depending on the dimensionality in velocity space, the 
  # distribution function is assigned accordingly:
  if(params.mode == '3V'):
    
    f = rho * (m/(2*np.pi*k*T_b))**(3/2) * \
        be.exp(-m*(p1 - p1_bulk)**2/(2*k*T_b)) * \
        be.exp(-m*(p2 - p2_bulk)**2/(2*k*T_b)) * \
        be.exp(-m*(p3 - p3_bulk)**2/(2*k*T_b))

  elif(params.mode == '2V'):

    f = rho * (m/(2*np.pi*k*T_b)) * \
        be.exp(-m*(p1 - p1_bulk)**2/(2*k*T_b)) * \
        be.exp(-m*(p2 - p2_bulk)**2/(2*k*T_b))

  else:

    f = rho *\
        np.sqrt(m/(2*np.pi*k*T_b)) * \
        be.exp(-m*(p1 - p1_bulk)**2/(2*k*T_b))

  return(f)