import numpy as np
from scorecond import scorecond
def estim_beta_pham(x):

    # X in shape of (N,T) where N is the number of variablse and T is the sample size.
    t1,t2 = x.shape
    if t1 > t2:
        print('error in eaastim_beta_pham(x): data must be organized in x in a row fashion')
        return 
    else:
        beta = np.zeros(x.shape)

        # For Test
        # t1 = np.array([    [2.9561  ,  2.1886 ,  -3.3958 ,  -4.4636  ,  2.7148], [-3.5623 ,  -2.6462 ,   4.6418  ,  5.2754 ,  -3.7086]])
        
        t1 = scorecond(x.T).T
        beta[0,:] = -t1[0,:]

        # For Test
        #t1 = np.array([[1.9600  , -2.6254  ,  0.7847  ,  1.1854 ,  -1.3048],[-6.6002 ,  11.7287 ,  -3.0391 ,  -4.7805 ,   2.6911]])
        # if x.shape[0]>1:
        t1 = scorecond(np.flipud(x).T).T
        beta = np.row_stack((beta, -t1[0,:]))

    

    return beta


# x = np.array([[0.2568,0.7799  ,  0.4925  ,  0.4762  ,  0.4906],
#         [0.2855  ,  0.7014  ,  0.9677  ,  0.9949  ,  0.5035]])
# beta = estim_beta_pham(x)
# print(beta)