import numpy as np
from pdinv import pdinv
from estim_beta_pham import estim_beta_pham
from adaptive_size import adaptive_size


def natural_grad_Adasize_Mask(x,Mask):

    N,T = x.shape
    mu = 1e-3
    itmax = 6000
    Tol = 1e-4
    Num_edges = np.sum(Mask)

    WW = np.eye(N,N)
    for i in range(N):
        Ind_i = np.nonzero(Mask[i]!=0)
        WW[i,Ind_i] = -0.5*np.matmul(np.matmul(x[i],x[Ind_i].T), pdinv(np.matmul(x[Ind_i],x[Ind_i].T)))
    
    W = 0.5*(WW+WW.T)

    z = np.zeros((N,N))
    eta = mu*np.ones(W.shape)
    W_old = W
    y_psi = [0 for i in range(N)]


    # use itera instead iter 
    y_psi0={}
    for itera in range(1,itmax):
        print('iteration :',itera,'/',itmax)
        y = np.matmul(W,x)

        if itera%12 == 1:
            for i in range(N):
                tem = estim_beta_pham(y[i].reshape(1,-1))
                y_psi[i] = tem[0]

                II = np.argsort(y[i])
                y_psi0[i] = y_psi[i][II]
        else:
            for i in range(N):

                II2 = np.argsort(y[i])
                y_psi[i][II2] = y_psi0[i]

        G = np.matmul(y_psi,y.T)/T
        yy = np.matmul(y,y.T)/T
        I_N = np.eye(N)

        Grad_W_n = np.matmul(y_psi,x.T)/T  + np.linalg.inv(W.T)
        if itera == 1:
            Grad_W_o = Grad_W_n
        
        eta,z = adaptive_size(Grad_W_n,Grad_W_o,eta,z)
        delta_W = eta*z
        W = W + delta_W*Mask

        if np.sum(np.abs(Grad_W_n*Mask))/Num_edges < Tol :
            print("early stop")
            break

        Grad_W_o = Grad_W_n
        W_old = W

    return W

# X = np.array([    
#     [0.1092  ,  0.6592,    0.6360,    0.0512,    0.2804],
#     [0.0066   , 0.5800,    0.5256,    0.7320,    0.2594],
#     [0.5973   , 0.9100,    0.2596,    0.1643,    0.5471]]) 
# Mask = np.array([
#         [1.0000,    0.4221,         0],
#         [0 ,   1.0000   ,      0],
#         [0  ,       0  ,  1.0000]
# ])

# print(natural_grad_Adasize_Mask(X,Mask))

