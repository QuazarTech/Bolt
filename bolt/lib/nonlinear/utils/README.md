# Utility Function for Nonlinear Solver:

This folder contains the utility functions that are used internally within the nonlinear solver and do not directly play a role in the computation. This folder contains the following files:

- `bandwidth_test.py`: Upon initialization of the nonlinear solver object we evaluate the bandwidth of the device being run on using the STREAMS benchmark which is printed along with the backend details.

- `broadcasted_primitive_operations.py`: In many of the functions in nonlinear/ we operate on arrays which are of different sizes. While one solution is to tile the arrays and perform the operation, a much cleaner implementation is to make use of the af.broadcast wrapped primitive functions such as addition and multiplication. af.broadcast allows us to perform batched operations on arrays of different sizes.

- `performance_timings.py`: This function prints the details of how much time has been spent inside each function along with the percentage of the total time spent in a nicely formatted table. Additionally this function also prints the number of zone-cycles per second. This function proves to be useful when analyzing performance characteristics and identifying bottlenecks.

- `print_with_indent.py`: This function is utilized when the nonlinear solver is initialized. This function is used to indent segments of the backend information to give a good formatted appearance.
