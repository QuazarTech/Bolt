cd collisional_damping
ipython run_lt.py
ipython run_ck.py
py.test

cd ../collisionless_damping
ipython run_lt.py
ipython run_ck.py
py.test

cd ../landau_damping
ipython run_lt.py
ipython run_ck.py
py.test

cd ../fields\ +\ collisions\ damping
ipython run_lt.py
ipython run_ck.py
py.test