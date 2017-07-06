import arrayfire as af
import numpy as np 

def f_maxwell_boltzmann(self, initial_parameters):

  m = initial_parameters.mass_particle
  k = initial_parameters.boltzmann_constant

  rho_b = initial_parameters.rho_background
  T_b   = initial_parameters.temperature_background

  p1_bulk = initial_parameters.p1_bulk_background
  p2_bulk = initial_parameters.p2_bulk_background
  p3_bulk = initial_parameters.p3_bulk_background

  pert_real = initial_parameters.pert_real
  pert_imag = initial_parameters.pert_imag

  k_q1 = initial_parameters.k_q1
  k_q2 = initial_parameters.k_q2

  p1, p2, p3 = self.p1_center, self.p2_center, self.p3_center

  # Calculating the perturbed density:
  rho = rho_b + (pert_real * af.cos(k_q1 * self.q1_center + k_q2 * self.q2_center) -\
                 pert_imag * af.sin(k_q1 * self.q1_center + k_q2 * self.q2_center)
                )

  # Depending on the dimensionality in velocity space, the 
  # distribution function is assigned accordingly:
  if(initial_parameters.mode == '3V'):
    
    f = rho * (m/(2*np.pi*k*T_b))**(3/2) * \
        af.exp(-m*(p1 - p1_bulk)**2/(2*k*T_b)) * \
        af.exp(-m*(p2 - p2_bulk)**2/(2*k*T_b)) * \
        af.exp(-m*(p3 - p3_bulk)**2/(2*k*T_b))

    f_background = rho_b * \
                   (m/(2*np.pi*k*T_b))**(3/2) * \
                    af.exp(-m*(p1 - p1_bulk)**2/(2*k*T_b)) * \
                    af.exp(-m*(p2 - p1_bulk)**2/(2*k*T_b)) * \
                    af.exp(-m*(p3 - p3_bulk)**2/(2*k*T_b))

  elif(initial_parameters.mode == '2V'):

    f = rho * (m/(2*np.pi*k*T_b)) * \
        af.exp(-m*(p1 - p1_bulk)**2/(2*k*T_b)) * \
        af.exp(-m*(p2 - p2_bulk)**2/(2*k*T_b))

    f_background = rho_b * \
                   (m/(2*np.pi*k*T_b)) * \
                    af.exp(-m*(p1 - p1_bulk)**2/(2*k*T_b)) * \
                    af.exp(-m*(p2 - p1_bulk)**2/(2*k*T_b))

  else:

    f = rho *\
        np.sqrt(m/(2*np.pi*k*T_b)) * \
        af.exp(-m*(p1 - p1_bulk)**2/(2*k*T_b))

    f_background = rho_b * \
                   np.sqrt(m/(2*np.pi*k*T_b)) * \
                   af.exp(-m*(p1 - p1_bulk)**2/(2*k*T_b))

  normalization = af.sum(f_background) * self.dp1 * self.dp2 * self.dp3/(f_background.shape[0])
  f             = (f/normalization)

  af.eval(f)
  return(f)