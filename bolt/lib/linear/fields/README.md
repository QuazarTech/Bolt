# Fields Solver:

This folder contains the routines that will be used when we need to couple a fields solver with the evolution of the linear solver. The fields solver routines are completely self-contained and only require the source terms such as charge density and currents to be inputed from the linear solver. This folder contains the following files:

- `fields_solver.py`: The file that contains the class which will be used to initialize a fields_solver object. The attributes of this class are initialized depending upon the input of the user, and are then evolved by the methods of the object.

- `electrostatic_solver.py`: We obtain electrostatic fields by solving the Poisson equation.
    
    If we assume that:

    delta_phi = delta_phi_hat * exp(1j * k_q1 * q1 + 1j * k_q2 * q2), 
    delta_rho = delta_rho_hat * exp(1j * k_q1 * q1 + 1j * k_q2 * q2), 

    then by substituting in the Poisson equation, we will get:
    
    (k_q1**2 + k_q2**2) * delta_phi_hat = delta_rho_hat 

- `dfields_hat_dt.py`: This file returns the dE1_hat/dt, dE2_hat/dt, dE3_hat/dt, dB1_hat/dt, dB2_hat/dt, dB3_hat/dt which is joined to give a single vector dfields_hat/dt. This is then passed to an integrator to evolve fields_hat. It is to be noted that f_hat, and fields_hat have to be at the same temporal location since they are coupled. For this purpose, coupled integrators have been defined under `integrators.py`.
