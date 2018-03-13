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

    f_initial = self.f
    self.f    = self.f + df_dt_fvm(self.f, self) * (dt / 2)
    
    self._communicate_f()
    self._apply_bcs_f()

    self.f = f_initial + df_dt_fvm(self.f, self) * dt

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
