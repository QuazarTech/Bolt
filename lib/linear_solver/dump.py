import h5py 

def dump_variables(self, file_name, *args):
  h5f = h5py.File(file_name + '.h5', 'w')
  for variable_name in args:
    h5f.create_dataset(str(variable_name), data = variable_name)
  h5f.close()
  return

def dump_distribution_function_5D(self, file_name):
  """
  Used to create the 5D distribution function array from the 3V delta_f_hat
  array. This will be used in comparison with the solution as given by the
  nonlinear method.
  """  
  h5f = h5py.File(file_name + '.h5', 'w')
  h5f.create_dataset('distribution_function', data = self.f)
  h5f.close()
  return