import numpy as np


def adaptive_size(grad_new, grad_old, eta_old, z_old):

    alpha = 0 
    up = 1.05
    down = 0.5

    z = grad_new + alpha * z_old

    etaup = (grad_new * grad_old) >= 0 # element-wise product

    eta = eta_old * (up * etaup + down * (1 - etaup))

    eta[eta>0.03] = 0.03

    return eta,z


# a = np.array([0.1537,0.8269,0.3010])
# eta,z=adaptive_size(a,a,a,a)
# print(eta,z)