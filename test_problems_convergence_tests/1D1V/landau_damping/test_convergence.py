import h5py
import numpy as np

def test_case():
  
  error     = np.zeros(5)
  N_x       = 2**(np.arange(5, 10))

  for i in range(len(N_x)):

    h5f  = h5py.File('distribution_function_data_files/lt/lt_distribution_function_' \
                      + str(N_x[i]) + '.h5', 'r'
                    )
    f_lt = h5f['distribution_function'][:]
    h5f.close()


    h5f  = h5py.File('distribution_function_data_files/ck/ck_distribution_function_' \
                      + str(N_x[i]) + '.h5', 'r'
                    )
    f_ck = h5f['distribution_function'][:]
    h5f.close()

    f_ck = np.swapaxes(f_ck, 0, 1).reshape(f_lt.shape[0], f_lt.shape[1], f_lt.shape[3], f_lt.shape[2])
    f_ck = np.swapaxes(f_ck, 3, 2)

    diff     = abs(f_ck - f_lt)
    error[i] = np.sum(diff)/f_ck.size

  poly = np.polyfit(np.log10(N_x), np.log10(error), 1)
  assert(abs(poly[0]+2)<0.2)