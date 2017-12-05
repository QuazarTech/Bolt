from .reconstruction_methods.minmod import reconstruct_minmod
from .reconstruction_methods.ppm import reconstruct_ppm
from .reconstruction_methods.weno5 import reconstruct_weno5

def reconstruct(self, input_array, axis, reconstruction_method):
    
    if(self.performance_test_flag == True):
        tic = af.time()

    if(reconstruction_method == 'piecewise-constant'):
        left_face_value  = input_array
        right_face_value = input_array

    elif(reconstruction_method == 'minmod'):
        left_face_value, right_face_value = reconstruct_minmod(input_array, axis)
        
    elif(reconstruction_method == 'ppm'):
        left_face_value, right_face_value = reconstruct_ppm(input_array, axis)

    elif(reconstruction_method == 'weno5'):
        left_face_value, right_face_value = reconstruct_weno5(input_array, axis)

    else:
        raise NotImplementedError('Reconstruction method invalid/not-implemented')
    
    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_reconstruct += toc - tic

    return(left_face_value, right_face_value)
