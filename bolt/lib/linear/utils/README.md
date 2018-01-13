# Utility Function for Nonlinear Solver:

This folder contains the utility functions that are used internally within the linear solver and do not directly play a role in the computation. This folder contains the following files:

- `bandwidth_test.py`: Upon initialization of the linear solver object we evaluate the bandwidth of the device being run on using the STREAMS benchmark which is printed along with the backend details.

- `broadcasted_primitive_operations.py`: In many of the functions in linear/ we operate on arrays which are of different sizes. While one solution is to tile the arrays and perform the operation, a much cleaner implementation is to make use of the af.broadcast wrapped primitive functions such as addition and multiplication. af.broadcast allows us to perform batched operations on arrays of different sizes.

- `fft_funcs.py`: Throughout Bolt, the common datastructure that is adopted is of shape [N_p, N_s, N_q1, N_q2, where the FFT operations need to be applied along N_q1 and N_q2. However, ArrayFire defaults to targeting the 1st two axes. For this purpose we define our own FFT functions which perform reorders on the data to get it to the required format, perform the FFTs and reorder back to the original format. 

- `print_with_indent.py`: This function is utilized when the linear solver is initialized. This function is used to indent segments of the backend information to give a nice well formatted appearance.
