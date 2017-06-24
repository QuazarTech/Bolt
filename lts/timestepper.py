def RK6_step(config, dY_dt, Y0, dt):
  """
  Evolves the various mode perturbation arrays by a single time-step by
  making use of the RK-6 time-stepping scheme:
  
  Parameters:
  -----------   
    config : Object config which is obtained by setup_simulation() is passed to 
             this file
 
    delta_f_hat : Array containing the values of the delta_f_hat at the time (t0 - dt).
                  (where t0 is the value as given by the time-stepping loop) 
 
    dt : Time-step size.

  Output:
  -------
    delta_f_hat : Array containing the values of the delta_f_hat at the time (t0).
                  (where t0 is the value as given by the time-stepping loop). 

  """

  k1 = dY_dt(config, Y0)
  k2 = dY_dt(config, Y0 + 0.25*k1*dt)
  k3 = dY_dt(config, Y0 + (3/32)*(k1+3*k2)*dt)
  k4 = dY_dt(config, Y0 + (12/2197)*(161*k1-600*k2+608*k3)*dt)
  k5 = dY_dt(config, Y0 + (1/4104)*(8341*k1-32832*k2+29440*k3-845*k4)*dt)
  k6 = dY_dt(config, Y0 + (-(8/27)*k1+2*k2-(3544/2565)*k3+(1859/4104)*k4-(11/40)*k5)*dt)

  Y_new = Y0 + 1/5*((16/27)*k1+(6656/2565)*k3+(28561/11286)*k4-(9/10)*k5+(2/11)*k6)*dt

  return(Y_new)

def RK4_step(config, dY_dt, Y0, dt):
  
  k1 = dY_dt(config, Y0)
  k2 = dY_dt(config, Y0 + 0.5*k1*dt)
  k3 = dY_dt(config, Y0 + 0.5*k2*dt)
  k4 = dY_dt(config, Y0 + 0.5*k3*dt)
  
  Y_new = Y0 + ((k1+2*k2+2*k3+k4)/6)*dt

  return(Y_new)