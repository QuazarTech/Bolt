import numpy as np
import params

moment_exponents = dict(density = [0, 0, 0],
                        j_x     = [1, 0, 0],
                        j_y     = [0, 1, 0]
                       )

moment_coeffs    = dict(density = [4./(2.*np.pi*params.h_bar)**2., 0, 0],
                        j_x     = [4./(2.*np.pi*params.h_bar)**2., 0, 0],
                        j_y     = [0, 4./(2.*np.pi*params.h_bar)**2., 0]
                       )
