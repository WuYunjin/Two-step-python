import numpy as np
from sparseica_W_adasize_Alasso_mask import sparseica_W_adasize_Alasso_mask
from betaAlasso_grad_2step import betaAlasso_grad_2step

def two_step_CD(X,lam):

    """
        % Two-step method for linear causal discovery that allows cycles and
        % confounderss
        % Input: 
        %   Data matrix X (variable number * sample size).
        % Output: 
        %   B: the causal influence matrix X = BX + E;
        %   in B causal direction goes from column to row, the entry Bij, means
        %   Xj -> Xi
        %   W_m: the ICA de-mixing matrix.
        % by Kun Zhang 2016,2017 Carnegie Mellon University
    """

    
    N,T = X.shape[0],X.shape[1]
    
    #   estimate the mask using adaptative lasso
    Mask = np.zeros((N,N))
    for i in range(N):
        if T<4*N:  # sample size too small, so preselect the features
            tmp1 = np.delete(X,i,axis=0) # get X\xi 
            Ind_t = np.argsort( np.abs(np.corrcoef(tmp1.T,X[i].T)))[::-1] # descend order # %compute the correlation of xi with X\xi and sort from larger in absolute value, get values and indices
            X_sel = tmp1[Ind_t[1:np.floor(N/4)]] #% pre-select N/4 features, the ones more correlated to xi
            
            beta_alt, beta_new_nt, beta2_alt, beta2_new_nt = betaAlasso_grad_2step(X_sel, X[i],0.65**2*np.var(X[i]), np.log(T)/2) 
            beta2_al = np.zeros(N-1)
            beta2_al[Ind_t[1:np.floor(N/4)]] = beta2_alt


        else:
            beta_al, beta_new_n, beta2_al, beta2_new_n = betaAlasso_grad_2step(np.delete(X,i,axis=0), X[i], 0.65**2*np.var(X[i]), np.log(T)/2)

        tmp = np.abs(beta2_al) >0.01
        Mask[i,0:i] = tmp[0:i]
        Mask[i,i+1:N] = tmp[i:N]

    Mask = Mask + Mask.T
    Mask = Mask!=0

    y_m, W_m, WW_m, Score = sparseica_W_adasize_Alasso_mask(np.log(T)*lam, Mask , X)
    B = np.eye(N) - W_m

    return B,W_m


if __name__ == "__main__":
    # data = np.loadtxt('E:\\code\\Two-Step-master\\Two-Step_Algorithm\\toy_example.txt')
    # B,W_m = two_step_CD(data,10)
    # print("B_estimate:\n",B)

    
    np.set_printoptions(precision=4)
    np.random.seed(1234)
    N = 5
    T = 1000
    data = np.random.uniform(size=(N,T))
    B = np.eye(N)

    B[0,1] = 0.2 + np.random.rand(1)
    B[1,2] = 0.2 + np.random.rand(1)
    B[2,4] = 0.2 + np.random.rand(1)

    E = np.random.uniform(size=(N,T))

    data = np.matmul(B,data)+E/100

    B_estimate,W_m = two_step_CD(data,30)

    print('B_truth:\n {}'.format(B))
    print('B_estimate:\n {}'.format(B_estimate))




    

