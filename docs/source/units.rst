*****
Units
*****

:math:`\texttt{Bolt}` handles dimensionless quantities normalized using some reference quantities. It's to be ensured that all input quantities are normalized appropriately when passing to ``physical_system``. This is to be done under the parameter and domain files under the respective problem folder.

Let us now illustrate this choice of normalization that can be adapted, when dealing with a purely collisonless case with electrostatic fields. Note that this is just one of the possible ways in which the independant units can be arrived at:

The equations governing by the system under consideration is given by:

.. math::
  \frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} + \frac{qE}{m} \frac{\partial f}{\partial v} = 0 \\
  E = -\frac{\partial \phi}{\partial x} \\
  \nabla^2 \phi = - \rho = - n q

- As we'd stated earlier, we have a choice of declaring a few variables with respect to absolute reference quantities(independant quantities), and find that the remaining dependant quantities can be expressed in terms of these. For this example, we'll choose appropriate reference units for time :math:`t`, velocity :math:`v`, charge :math:`q` and mass :math:`m`.

- When dealing with a plasma at constant mean density :math:`n_e` , it is convenient to normalize times by introducing the electron plasma frequency :math:`\omega_{pe} = \sqrt{\frac{ne^2}{m}}`. Then all times are normalized using :math:`{\omega_{pe}}^{-1}`. For the sake of convenience, let's call this normalization factor :math:`t_0` (where :math:`t_0 = {\omega_{pe}}^{-1}`):

.. math::
    t = t_0 \bar{t}

- Equating the kinetic energy and thermal energy of the plasma, we obtain the thermal velocity which we use for scaling the velocity terms:

.. math::
    \frac{1}{2} m {v_0}^2 = \frac{1}{2} k T \implies v_0 = \sqrt{\frac{k T}{m}}

So the velocity can be expressed as:

.. math::
    v = v_0 \bar{v}

- Expressing the charge and the mass interms of our reference units :math:`e_0` and :math:`m_0` which are typically taken as the electron charge and mass:

.. math::
    q = e_0 \bar{q} \\
    m = m_0 \bar{m}

Now substituting these back into Vlasov-Boltzmann equation, we get:

.. math::
  \frac{1}{t_0} \frac{\partial f}{\partial \bar{t}} + v_0 \bar{v} \frac{\partial f}{\partial x} + \frac{e_0}{m_0 v_0} \frac{\bar{q}E}{\bar{m}} \frac{\partial f}{\partial \bar{v}} = 0 \\
  \implies \frac{\partial f}{\partial \bar{t}} + v_0 t_0 \bar{v} \frac{\partial f}{\partial x} + \frac{e_0 t_0}{m_0 v_0} \frac{\bar{q}E}{\bar{m}} \frac{\partial f}{\partial \bar{v}} = 0 \\
  \implies \frac{\partial f}{\partial \bar{t}} + \bar{v} \frac{\partial f}{\partial (\frac{x}{v_0 t_0})} +  \frac{\bar{q}}{\bar{m}} \frac{E}{(\frac{m_0 v_0}{e_0 t_0})} \frac{\partial f}{\partial \bar{v}} = 0 \\
  \implies \frac{\partial f}{\partial \bar{t}} + \bar{v} \frac{\partial f}{\partial \bar{x}} +  \frac{\bar{q} \bar{E}}{\bar{m}} \frac{\partial f}{\partial \bar{v}} = 0

Thus, we find that the normalization constant for the distance :math:`x` and electric field :math:`E` come out in terms of the independantly chosen references:

.. math::
    x = x_0 \bar{x}; where\ x_0 = v_0 t_0 \\
    E = E_0 \bar{E}; where\ E_0 = \frac{m_0 v_0}{e_0 t_0}

Now, let's take a look at the appropriate normalizations that need to be applied for the electric potential:

.. math::
  E = -\frac{\partial \phi}{\partial x} \\
  \implies E_0 \bar{E} = -\frac{1}{x_0} \frac{\partial \phi}{\partial \bar{x}} \\
  \implies \bar{E} = -\frac{\partial (\frac{\phi}{E_0 x_0})}{\partial \bar{x}} \\
  \implies \bar{E} = -\frac{\partial \bar{\phi}}{\partial \bar{x}}

Hence, we get :math:`\bar{\phi} = E_0 x_0 = \frac{m_0 v_0^2}{e_0 t_0}`

The table below gives a list of the normalizations we had used in this case, clearly distinguishing between the dependant and the independant quantites:

**Independant Quantities**:

+--------------------+------------------+
|Physical Quantity   | Reference Unit   | 
+====================+==================+ 
| Time               | :math:`t_0`      | 
+--------------------+------------------+ 
| Velocity           | :math:`v_0`      | 
+--------------------+------------------+ 
| Charge             | :math:`e_0`      | 
+--------------------+------------------+
| Mass               | :math:`m_0`      | 
+--------------------+------------------+

**Dependant Quantities**:

+--------------------+----------------------------------+
|Physical Quantity   | Reference Unit                   | 
+====================+==================================+ 
| Distance           | :math:`v_0 t_0`                  | 
+--------------------+----------------------------------+ 
| Electric Field     | :math:`\frac{m_0 v_0}{e_0 t_0}`  | 
+--------------------+----------------------------------+ 
| Electric Potential | :math:`\frac{m_0 v_0^2}{e_0 t_0}`| 
+--------------------+----------------------------------+
