init:
	- pip install -r requirement.txt
	- export PYTHONPATH=$PWD:$PYTHONPATH

test:
	- py.test bolt/lib/linear_solver/tests
	- py.test bolt/lib/nonlinear_solver/tests
