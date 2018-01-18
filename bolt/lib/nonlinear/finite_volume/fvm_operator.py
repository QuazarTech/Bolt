import arrayfire as af
from .df_dt_fvm import df_dt_fvm
from bolt.lib.nonlinear.utils.broadcasted_primitive_operations import multiply

def op_fvm(self, dt):
    """
    Evolves the system defined using FVM. It does so by integrating 
    the function df_dt using an RK2 stepping scheme. After the initial 
    evaluation at the midpoint, we evaluate the currents(J^{n+0.5}) and 
    pass it to the FDTD algo when an electrodynamic case needs to be evolved.
    The FDTD algo updates the field values, which are used at the next
    evaluation of df_dt.

    Parameters
    ----------

    dt : double
         Time-step size to evolve the system
    """

    self._communicate_f()
    self._apply_bcs_f()

    if(self.performance_test_flag == True):
        tic = af.time()

    # Solving for tau = 0 systems:
    tau = self.physical_system.params.tau(self.q1_center, self.q2_center,
                                          self.p1_center, self.p2_center, 
                                          self.p3_center
                                         )
    if(af.any_true(tau == 0)):
        
        self.f = af.select(tau == 0, 
                           self._source(self.f, self.time_elapsed,
                                        self.q1_center, self.q2_center,
                                        self.p1_center, self.p2_center, self.p3_center, 
                                        self.compute_moments, 
                                        self.physical_system.params, 
                                        True
                                       ),
                           self.f
                          )

    f_initial = self.f
    self.f    = self.f + df_dt_fvm(self.f, self) * (dt / 2)

    self._communicate_f()
    self._apply_bcs_f()

    if(self.physical_system.params.EM_fields_enabled == True):
        
        # Evolving electrodynamic fields:
        if(self.physical_system.params.fields_solver == 'fdtd'):
            
            J1 = multiply(self.physical_system.params.charge, 
                          self.compute_moments('mom_v1_bulk')
                         )  # (i + 1/2, j + 1/2)
            J2 = multiply(self.physical_system.params.charge, 
                          self.compute_moments('mom_v2_bulk')
                         )  # (i + 1/2, j + 1/2)
            J3 = multiply(self.physical_system.params.charge, 
                          self.compute_moments('mom_v3_bulk')
                         )  # (i + 1/2, j + 1/2)

            self.fields_solver.evolve_electrodynamic_fields(J1, J2, J3, dt)

        # Since it will be evaluated again at the midpoint
        if(self.physical_system.params.fields_type == 'user-defined'):
            self.time_elapsed += 0.5 * dt

    self.f = f_initial + df_dt_fvm(self.f, self) * dt

    if(self.physical_system.params.EM_fields_enabled == True):
        # Subtracting the change made to avoid messing 
        # with the counter on timestep.py
        if(self.physical_system.params.fields_type == 'user-defined'):
            self.time_elapsed -= 0.5 * dt

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fvm_solver += toc - tic
    
    af.eval(self.f)
    return
