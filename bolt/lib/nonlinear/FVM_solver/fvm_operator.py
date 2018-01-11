import arrayfire as af
from .timestep_df_dt import fvm_timestep_RK2

def op_fvm(self, dt):
    """
    Evolves the system defined using FVM.

    Parameters
    ----------

    dt : float
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
                                        self.p1, self.p2, self.p3, 
                                        self.compute_moments, 
                                        self.physical_system.params, 
                                        True
                                       ),
                           self.f
                          )

    fvm_timestep_RK2(self, dt)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_fvm_solver += toc - tic
    
    af.eval(self.f)
    return
