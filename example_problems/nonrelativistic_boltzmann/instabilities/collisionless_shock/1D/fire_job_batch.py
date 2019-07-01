import os
import numpy as np

tau_start = 0.1  #* t0
tau_end   = 100. #* t0
tau_step  = 10.  #* t0

source_filepath = \
        '/home/mchandra/bolt_master/bolt/example_problems/nonrelativistic_boltzmann/instabilities/collisionless_shock/1D'
job_script_file = 'job_script_gpu3001' # Can also be empty
run_filepath = \
        '/home/mchandra/bolt_master/bolt/example_problems/nonrelativistic_boltzmann/instabilities/collisionless_shock/1D/tau_'

submit_jobs = False # Be careful!

for tau in np.arange(tau_start, tau_end, tau_step):
    filepath = run_filepath + str(tau)

    # If folder does not exist, make one and add all files from source folder

    if not os.path.isdir(filepath):
        os.makedirs(filepath)
        os.makedirs(filepath+"/dump_f")
        os.makedirs(filepath+"/dump_fields")
        os.makedirs(filepath+"/dump_moments")
        os.makedirs(filepath+"/images")

        os.system("cp " + (source_filepath + "/*.py ") 
                        + (source_filepath + "/" + job_script_file)
                        + " " + filepath   + "/."
                 )
       
        # Change required files
        # Code copied from here : 
        # https://stackoverflow.com/questions/4454298/prepend-a-line-to-an-existing-file-in-python
        
        f = open(filepath + "/params.py", "r")
        old = f.read() # read everything in the file
        f.close() 
        
        f = open(filepath + "/params.py", "w")
        f.write("tau_collisions = " + str(tau) + " \n")
        f.write(old)
        f.close()

    # Schedule job after changing to run directory so that generated slurm file
    # is stored in that directory
    os.chdir(filepath)

    if (submit_jobs):
        os.system("sbatch " + job_script_file)

    os.chdir(source_filepath) # Return to job firing script's directory
