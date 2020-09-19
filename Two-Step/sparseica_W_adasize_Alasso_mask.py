import numpy as np 

from natural_grad_Adasize_Mask import natural_grad_Adasize_Mask
from estim_beta_pham import estim_beta_pham
from adaptive_size import adaptive_size

def sparseica_W_adasize_Alasso_mask(lam, Mask, x):

    
    N,T = x.shape
    xx = x - np.mean(x.T,axis=0).reshape(-1,1)
    # % % To avoid instability
    xx = np.matmul(np.diag(1./np.std(xx.T,axis=0)) , xx)
    Refine = 1
    Num_edges = np.sum(Mask)
        
    # % learning rate
    mu = 1E-3 # % 1E-6
    beta = 0 # % 1
    save_intermediate = 0
    m = 60# % for approximate the derivative of |.|
    # % a = 3.7;
    itmax = 10000
    iter_M = 200
    delta_H = 0
    Tol = 1e-4
    w11_back = []
    w12_back = []
    W_backup = np.zeros((N,N))
    eta_backup = np.zeros((N,N))
    z_backup = np.zeros((N,N))
    grad_backup = np.zeros((N,N))

        
    # % initiliazation
    print('Initialization....\n')
    WW = np.diag(1./np.std(xx.T,axis=0))
    WW = natural_grad_Adasize_Mask(xx, Mask)

    omega1 = 1/np.abs((WW.T[Mask!=0]))
        
    # % to avoid instability
    Upper = 3 * np.mean(omega1)
    omega1[omega1>Upper] = Upper

    
    omega = np.zeros((N,N))
    omega[Mask!=0] = omega1
    omega = omega.T
    W = WW
        
    z = np.zeros((N,N))
    eta = mu * np.ones(W.shape)
    W_old = W + np.eye(N)
    grad_new = W_old
    y_psi = [0 for i in range(N)]
    y_psi0 = {}

        
    print('Starting penalization...\n')
    for itera in range(1,itmax):
        print('iteration :',itera,'/',itmax)
        y = np.matmul(W,xx)
        if np.sum(np.abs(grad_new* Mask))/Num_edges<Tol:
            print("early stop")
            if Refine:
                Mask = np.abs(W) > 0.02
                Mask = Mask^np.diag(np.diag(Mask))
                lam = 0
                Refine = 0
            else:
                break
        
        W_old = W 

        
        # % update W: linear ICA with marginal score function estimated from data...
     
        if itera%12 ==1 :
            for i in range(N):
                tem = estim_beta_pham(y[i].reshape(1,-1))
                y_psi[i] = tem[0]
                II = np.argsort(y[i])
                y_psi0[i] = y_psi[i][II]
            
        else:
            for i in range(N):
                II2 = np.argsort(y[i])
                y_psi[i][II2] = y_psi0[i]

        dev = omega*np.tanh(m*W)
        grad_new = np.matmul(y_psi,x.T)/T + np.linalg.inv(W.T) - 4*beta* np.matmul( np.diag(np.diag(np.matmul(y,y.T)/T)) - np.eye(N) ,np.matmul(y,x.T)/T) - dev*lam/T

        if itera == 1:
            grad_old = grad_new

        
        # % adaptive size
        eta, z = adaptive_size(grad_new, grad_old, eta, z)
        delta_W = eta*z
        W = W + 0.9* delta_W * Mask

        grad_old = grad_new

       
        # if save_intermediate
        #     W_backup(:,:,iter) = W;
        #     z_backup(:,:,iter) = z;
        #     eta_backup(:,:,iter) = eta;
        #     grad_backup(:,:,iter) = grad_new;
        # end

    Score = omega * np.abs(W)

    return y, W, WW, Score