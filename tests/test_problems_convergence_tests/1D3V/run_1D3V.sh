cd collisional_damping
ipython run_linear_solver.py
ipython run_non_linear_solver.py
py.test

cd ../collisionless_damping
ipython run_linear_solver.py
ipython run_non_linear_solver.py
py.test

cd ../landau_damping
ipython run_linear_solver.py
ipython run_non_linear_solver.py
py.test

cd ../fields\ +\ collisions\ damping
ipython run_linear_solver.py
ipython run_non_linear_solver.py
py.test