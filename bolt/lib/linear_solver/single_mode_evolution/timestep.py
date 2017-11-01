def RK4(self, dt):

    Y_initial = self.Y.copy()

    k1 = self._dY_dt()
    x  = Y_initial + 0.5 * k1 * dt
    k2 = self._dY_dt()
    x  = Y_initial + 0.5 * k2 * dt
    k3 = self._dY_dt()
    x  = Y_initial + k3 * dt
    k4 = self._dY_dt()

    self.Y = Y_initial + ((k1 + 2 * k2 + 2 * k3 + k4) / 6) * dt

    return
