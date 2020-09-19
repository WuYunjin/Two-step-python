import numpy as np

def pdinv(A):

    # PDINV Computes the inverse of a positive definite matrix
    try:
        Ainv = np.linalg.inv(A)

    except:
        print('Matrix is not positive definite in pdinv, inverting, using svd')
        Ainv = np.linalg.pinv(A)

    return Ainv


# A = np.array([
# [    0.6356 ,  -0.1960   , 0.1036],
# [   -0.1960  ,  0.7207   , 0.2317],
# [    0.1036   , 0.2317  ,  0.2099]])

# print(pdinv(A))