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

    self._communicate_f()
    self._apply_bcs_f()

    f_initial = self.f # this is f^{n}
    self.f    = self.f + df_dt_fvm(self.f, self) * (dt / 2) # this is f{n+1/2}
    
    self._communicate_f()
    self._apply_bcs_f()

    # this is equivalent to f^{n+1} = f^n + df_dt(f = f^{n+1/2}) * dt
    # there df_dt() is the function that returns df / dt
    self.f = f_initial + df_dt_fvm(self.f, self) * dt

    return

def op_fvm(self, dt):

    if(self.performance_test_flag == True):
        tic = af.time()

    timestep_fvm(self, dt)

    af.eval(self.f)
    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fvm_solver += toc - tic
    
    return
