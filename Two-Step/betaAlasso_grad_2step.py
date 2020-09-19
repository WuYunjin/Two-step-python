import numpy as np
from pdinv import pdinv

def betaAlasso_grad_2step(x,y,var_noise,lam):
    """
    % Aim: 
    %       to find the solution of adaptive Lasso with the given lambda
    % Inputs: 
    %       x is p*n, and y is 1*n. var_noise is the variance of the noise.
    % Outputs: 
    %       beta_al contains the obtained beta after step 1 (conventioanl
    % ALasso). After convergence of step 1, we repeat upadte \hat{beta} and
    % repeat the adaptive Lasso procedure to refine the result.
    %       beta_new_n contains the penalty term (beta_al/\hat{beta}) after
    %       step 1.
    %       beta2_al and beta2_new_n contain the results after step 2.
    % Note that in the first step, a combination of gradient-based methods and 
    % Newton method is used.
    % by Kun Zhang 2016,2017 Carnegie Mellon University
    """
    
    var_noise_back = var_noise
    Trad1 = 0.2
    Trad2 = 1 - Trad1
    N,T = x.shape
    tol = 1E-2
    beta_min = 1E-12
    beta_min2 = 1E-2
    sum_adjust_beta = []
    pl = []

    beta_hat = np.matmul(np.matmul(y,x.T), np.linalg.inv(np.matmul(x,x.T)))
    if var_noise == 0:
        var_noise = np.var(y- np.matmul(beta_hat,x))

    x_new = np.matmul(np.diag(beta_hat),x) 
    Error = 1
    beta_new_o = np.ones(N)

    # % store for curve plotting
    # sum_adjust_beta = [sum_adjust_beta sum(abs(beta_new_o))];
    # pl = [pl (y-beta_new_o'*x_new)*(y-beta_new_o'*x_new)'/2/var_noise + lambda * sum(abs(beta_new_o))];

    
    while Error > tol :
        Sigma = np.diag(1/np.abs(beta_new_o))
        beta_new_n = np.matmul(np.linalg.inv( np.matmul(x_new,x_new.T)+ var_noise*lam*Sigma ), np.matmul(x_new,y.T))*Trad1 + beta_new_o*Trad2
        beta_new_n= np.sign(beta_new_n)*np.maximum(np.abs(beta_new_n),beta_min)
        Error = np.linalg.norm(beta_new_n - beta_new_o)
        beta_new_o = beta_new_n
        
        # sum_adjust_beta = [sum_adjust_beta sum(abs(beta_new_n))];
        # pl = [pl (y-beta_new_n'*x_new)*(y-beta_new_n'*x_new)'/2/var_noise + lambda * sum(abs(beta_new_n))];

    
    Ind = np.nonzero(np.abs(beta_new_n)>1e4*beta_min)
    beta_new_n = beta_new_n * (np.abs(beta_new_n)>1e4*beta_min)

    beta_al = beta_new_n * beta_hat.T


    #step 2
    N2 = len(Ind[0])
    x2 = x[Ind]
    beta2_hat = np.matmul(np.matmul(y.reshape(1,-1),x2.T), np.linalg.inv(np.matmul(x2,x2.T)) )
    if var_noise_back == 0 :
        var_noise = np.var(y- np.matmul(beta2_hat[0],x2))

    x2_new = np.matmul(np.diag(beta2_hat[0]), x2)
    beta2_new_o = np.ones(N2)
    # sum_adjust_beta2 = []
    # pl2 = [];    
    # sum_adjust_beta2 = [sum_adjust_beta2 sum(abs(beta2_new_o))];
    # pl2 = [pl2 (y-beta2_new_o'*x2_new)*(y-beta2_new_o'*x2_new)'/2/var_noise + lambda * sum(abs(beta2_new_o))];

    Error = 1
    Iter = 1
    while Error > tol :
        Sigma = np.diag(1/np.abs(beta2_new_o))
        # if det(x2_new*x2_new' + var_noise*lambda * Sigma) < 0.01
        #     pause;
        # end

        beta2_new_n = np.matmul(pdinv(  np.matmul(x2_new,x2_new.T) + var_noise*lam*Sigma) , np.matmul(x2_new,y.T))

        beta2_new_n = np.sign(beta2_new_n)*np.maximum(np.abs(beta2_new_n),beta_min)
        Error = np.linalg.norm(beta2_new_n - beta2_new_o)
        beta2_new_o = beta2_new_n

        # sum_adjust_beta2 = [sum_adjust_beta2 sum(abs(beta2_new_n))];
        # pl2 = [pl2 (y-beta2_new_n'*x2_new)*(y-beta2_new_n'*x2_new)'/2/var_noise + lambda * sum(abs(beta2_new_n))];

        Iter = Iter + 1
        if Iter > 100:
            break

    
    beta2_new_n = beta2_new_n* (np.abs(beta2_new_n)>beta_min2)

    beta2_al = np.zeros(N)

    beta2_al[Ind] = beta2_new_n * beta2_hat[0].T

    return beta_al, beta_new_n, beta2_al, beta2_new_n
    