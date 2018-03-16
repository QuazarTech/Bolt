import arrayfire as af
from .df_dt_fvm import df_dt_fvm
from bolt.lib.utils.broadcasted_primitive_operations import multiply
from bolt.lib.nonlinear.temporal_evolution import operator_splitting_methods as split

def timestep_fvm(self, dt):
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
    # self.fields_solver.check_maxwells_contraint_equations(-10 * self.compute_moments('density'))

    # rho_n       = -1 * self.compute_moments('density')
    # rho_n[0, 1] = -1 * rho_n[0, 1]
    # rho_n       = af.sum(rho_n, 1)

    f_initial = self.f
    self.f    = self.f + df_dt_fvm(self.f, self) * (dt / 2)
    
    self._communicate_f()
    self._apply_bcs_f()

    self.f = f_initial + df_dt_fvm(self.f, self) * dt

    # self.count += 1

    # rho_n_plus_one       = -1 * self.compute_moments('density')
    # rho_n_plus_one[0, 1] = -1 * rho_n_plus_one[0, 1]
    # rho_n_plus_one       = af.sum(rho_n_plus_one, 1)

    # drho_dt = (rho_n_plus_one - rho_n) / dt
    
    # J1 = self.fields_solver.J1
    # J2 = self.fields_solver.J2

    # J1_plus_q1 = af.shift(self.fields_solver.J1, 0, 0, -1)
    # J2_plus_q2 = af.shift(self.fields_solver.J2, 0, 0, 0, -1)

    # divJ = (J1_plus_q1 - J1) / self.dq1 + (J2_plus_q2 - J2) / self.dq2

    # print(af.mean(af.abs(drho_dt + divJ)[:, :, 4:-4, 4:-4]))

    # print(af.sum(af.abs(af.sum(((self.f - f_initial)*0 / dt + 
    #                                 0*multiply(self.p1_center, (af.shift(self.f_left, 0, 0, -1) - self.f_left) / self.dq1) +
    #                                 multiply(-10 * self.fields_solver.cell_centered_EM_fields[0], (self.f_right_p1 - self.f_left_p1) / self.dp1)), 1
    #                           )
    #                    )
    #             )
    #      ) 

    # self.data[self.count] = af.sum(af.abs(af.sum(((self.f - f_initial) / dt + 
    #                                 multiply(self.p1_center, (af.shift(self.f_left, 0, 0, -1) - self.f_left) / self.dq1) +
    #                                 multiply(-10 * self.fields_solver.cell_centered_EM_fields_at_n[0], (self.f_right_p1 - self.f_left_p1) / self.dp1)), 0
    #                           )
    #                    )
    #             )

    # print(self.data[self.count])
    # print(af.sum(af.abs(multiply(-10 * self.fields_solver.cell_centered_EM_fields_at_n[0], (self.f[-1] - self.f[0]) / self.dp1))))

def update_for_instantaneous_collisions(self, dt):
    
    self.f = self._source(self.f, self.time_elapsed,
                          self.q1_center, self.q2_center,
                          self.p1_center, self.p2_center, self.p3_center, 
                          self.compute_moments, 
                          self.physical_system.params, 
                          True
                         )

    return

def op_fvm(self, dt):

    self._communicate_f()
    self._apply_bcs_f()

    if(self.performance_test_flag == True):
        tic = af.time()

    if(self.physical_system.params.instantaneous_collisions == True):
        split.strang(self, timestep_fvm, update_for_instantaneous_collisions, dt)
    else:
        timestep_fvm(self, dt)

    af.eval(self.f)
    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fvm_solver += toc - tic
    
    return
