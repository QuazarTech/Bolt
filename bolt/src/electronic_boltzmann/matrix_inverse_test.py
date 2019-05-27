import numpy as np
import arrayfire as af

def inverse_3x3_matrix(A):
# TO TEST:
#        A_test     = np.random.rand(3, 3)
#        A_inv_test = np.linalg.inv(A_test)
#        A_inv      = np.array(inverse_3x3_matrix(A_test))
#        print("err = ", np.max(np.abs(A_inv - A_inv_test)))


    det = \
        A[0][0]*A[1][1]*A[2][2] \
      - A[0][0]*A[1][2]*A[2][1] \
      - A[0][1]*A[1][0]*A[2][2] \
      + A[0][1]*A[1][2]*A[2][0] \
      + A[0][2]*A[1][0]*A[2][1] \
      - A[0][2]*A[1][1]*A[2][0] \

    #af.eval(det)

    #TODO : Raise an exception if the matrix is singular
    #print ('determinant : ')
    #print (det)

    A_inv = [[0, 0, 0], \
             [0, 0, 0], \
             [0, 0, 0] \
            ]

    cofactors = [[0, 0, 0], \
                 [0, 0, 0], \
                 [0, 0, 0] \
                ]
    
    adjoint = [[0, 0, 0], \
               [0, 0, 0], \
               [0, 0, 0] \
              ]

    cofactors[0][0] = +(A[1][1]*A[2][2] - A[1][2]*A[2][1])
    cofactors[0][1] = -(A[1][0]*A[2][2] - A[1][2]*A[2][0])
    cofactors[0][2] = +(A[1][0]*A[2][1] - A[1][1]*A[2][0])
    cofactors[1][0] = -(A[0][1]*A[2][2] - A[0][2]*A[2][1])
    cofactors[1][1] = +(A[0][0]*A[2][2] - A[0][2]*A[2][0])
    cofactors[1][2] = -(A[0][0]*A[2][1] - A[0][1]*A[2][0])
    cofactors[2][0] = +(A[0][1]*A[1][2] - A[0][2]*A[1][1])
    cofactors[2][1] = -(A[0][0]*A[1][2] - A[0][2]*A[1][0])
    cofactors[2][2] = +(A[0][0]*A[1][1] - A[0][1]*A[1][0])

    adjoint[0][0] = cofactors[0][0]
    adjoint[0][1] = cofactors[1][0]
    adjoint[0][2] = cofactors[2][0]
    adjoint[1][0] = cofactors[0][1]
    adjoint[1][1] = cofactors[1][1]
    adjoint[1][2] = cofactors[2][1]
    adjoint[2][0] = cofactors[0][2]
    adjoint[2][1] = cofactors[1][2]
    adjoint[2][2] = cofactors[2][2]


    A_inv = adjoint/det
  
    #A_inv[0][1] = adjoint[0][1]/det
  
    #A_inv[0][2] = adjoint[0][2]/det
  
    #A_inv[1][0] = adjoint[1][0]/det
    #
    #A_inv[1][1] = adjoint[1][1]/det
  
    #A_inv[1][2] = adjoint[1][2]/det
  
    #A_inv[2][0] = adjoint[2][0]/det

    #A_inv[2][1] = adjoint[2][1]/det
  
    #A_inv[2][2] = adjoint[2][2]/det
  

    arrays_to_be_evaled = \
        [A_inv[0][0], A_inv[0][1], A_inv[0][2], \
         A_inv[1][0], A_inv[1][1], A_inv[1][2], \
         A_inv[2][0], A_inv[2][1], A_inv[2][2] \
        ]

    #af.eval(*arrays_to_be_evaled)

    return(A_inv)
def main():
    A_test     = np.random.rand(3, 3)
    A_inv_test = np.linalg.inv(A_test)
    A_inv      = np.array(inverse_3x3_matrix(A_test))
    print("err = ", A_inv - A_inv_test)

main()
