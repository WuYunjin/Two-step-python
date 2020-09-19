import numpy as np
import math
import scipy.sparse as sps


# np.set_printoptions(precision=4)

def scorecond(data, q=None, bdwidth=None, cova=None):

    """     
            Estimate the conditional score function defined as minus the
            gradient of the conditional density of of a random variable X_p
            given x_{p-1}, \dots, x_{q+1}. The values of these random variables are
            provided in the n x p array data.
            
            The estimator is based on the partial derivatives of the
            conditional entropy with respect to the data values, the entropy
            being estimated through the kernel density estimate _after a
            prewhitening operation_, with the kernel being the density of the
            sum of 3 independent uniform random variables in [.5,.5]. The kernel
            bandwidth is set to bdwidth*(standard deviation) and the density is
            evaluated at bandwidth apart. bdwidth defaults to
                2*(11*sqrt(pi)/20)^((p-q)/(p-q+4))*(4/(3*n))^(1/(p-q+4)
            (n = sample size), which is optimal _for estimating a normal density_
            
            If cova (a p x p matrix) is present, it is used as the covariance
            matrix of the data and the mean is assume 0. This prevents
            recomputation of the mean and covariance if the data is centered
            and/or prewhitenned.
            
            The score function is computed at the data points and returned in
            psi.

    """
    if data is None:
        print('usage: psi = scorecond(data [, bdwidth, cova])')
        return

    
    n, p = data.shape
    if q is None:
        q = 0

    if p<q+1:
        print('Sorry: not enough variables')
        return

    if cova is None:
        tmp = np.mean(data,axis=0) # mean of data
        data = data - tmp # centered data
        cova = np.matmul(data.T,data)/n; # covariance matrix

    
    T = np.linalg.cholesky(cova).T ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    data = np.matmul(data,np.linalg.inv(T)) # i.e. data/T

        
    if q > 0:
        data[:,0:q] = []# delete first q columns
        p = p-q

    
    if bdwidth is None:
        bdwidth = 2* (11*math.sqrt(math.pi)/20)**(p/(p+4)) * (4/(3*n))**(1/(p+4))
    
    
    # Grouping the data into cells, idx gives the index of the cell
    # containing a datum, r gives its relative distance to the leftmost
    # border of the cell

        
    r = data/bdwidth
    idx = np.floor(r)
    r = r - idx
    tmp = np.min(idx,axis=0)
    idx = idx - tmp*np.ones((n,1)) + 1 # 0 <= idx-1 



    # Compute the probabilities at grid cells
    # The kernel function is
    #
    #        1/2 + (1/2+u)(1/2-u) |u| <= 1/2
    # k(u) = (3/2 - |u|)^2/2 1/2 <= |u| <= 3/2
    #        0 otherwise
    #
    # The contribution to the probability at i-1, i, i+1 by a
    # data point at distance r (0 <= r < 1) to i, are respectively:
    # (1-r)^2/2, 1/2 + r*(1-r), r^2/2
    # The derivative of the contribution to the probability at i-1, i, i+1
    # by a data point at distance r to i are respectively: r-1, 1-2r, r

    # The array ker contains the contributions to the probability of cells
    # The array kerp contains the gradient of these contributions
    # The array ix contains the indexes of the cells, arranged in
    # _lexicographic order_

    ker = np.array([(1-r[:,0])**2/2, .5 + r[:,0]*(1-r[:,0]),r[:,0]**2/2]).T
    ix = np.array([idx[:,0] ,idx[:,0]+1,idx[:,0]+2]).T
    kerp = np.array([1-r[:,0],2*r[:,0]-1,-r[:,0]]).T
    mx = np.max(idx,axis=0) + 2
    M = np.cumprod(mx)
    nr = np.array(range(0,n))

    for i in range(1,p):
        ii = 1*np.ones((1,3**(i)))#i*np.ones((1,3**(i)))
        kerp = np.concatenate([np.concatenate([ kerp* ((1-r[nr,i].reshape(-1,1)*ii )**2)/2,   #r[nr,i].reshape(-1,1)*ii is r(nr,ii) in the original matlab code
                            kerp*(.5 + (r[nr,i].reshape(-1,1)*ii)*(1-r[nr,i].reshape(-1,1)*ii)),
                            kerp* ((r[nr,i].reshape(-1,1)*ii)**2)/2],axis=1),
                            np.concatenate( [ker*(1-(r[:,i].reshape(-1,1)*ii)), ker*(2*(r[:,i].reshape(-1,1)*ii)-1), -ker*(r[:,i].reshape(-1,1)*ii) ],axis=1) #(r[:,i-1].reshape(-1,1)*ii) is r(:,ii) in the original matlab code
        ])

        nr = np.concatenate([nr,np.array(range(0,n))]) #= 1:n repeated i times

        ker = np.concatenate([ ker* ((1-(r[:,i].reshape(-1,1)*ii))**2)/2,
                                ker*(.5 + (r[:,i].reshape(-1,1)*ii)*(1- (r[:,i].reshape(-1,1)*ii) )),
                                ker* ((r[:,i].reshape(-1,1)*ii) **2)/2
        ],axis=1)

        Mi = M[i-1]

        #(idx[:,i].reshape(-1,1)*ii) is idx(:,ii) in the original matlab code
        ix = np.concatenate([ix+Mi*((idx[:,i].reshape(-1,1)*ii)-1), ix+Mi*(idx[:,i].reshape(-1,1)*ii), ix+Mi*((idx[:,i].reshape(-1,1)*ii)+1)],axis=1)
        
        ix = ix - 1  # Because the python index starts from 0

    ###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    pr = sps.csc_matrix(( ker.reshape(-1), (ix.astype(int).reshape(-1),np.array([0]*len(ix.reshape(-1))))),shape=(1+M[p-1].astype(int),1),dtype=np.float)/n # joint prob. of cells

    logp = sps.csc_matrix(np.zeros(shape=(1+M[p-1].astype(int),1)),shape=(1+M[p-1].astype(int),1))

    if p > 1:
        pm = np.sum(pr.reshape(Mi.astype(int),mx[p-1].astype(int),order='F'), axis=1) # marginal prob. (Mi = M(p-1))
        pm =  (pm*np.ones((1,mx[p-1].astype(int)))).reshape(1+M[p-1].astype(int),1,order='F')
        logp[np.nonzero(pr)] =  np.log(pr[np.nonzero(pr)] /pm[np.nonzero(pr)] ) # avoid 0
    else:
        logp[np.nonzero(pr)] =  np.log(pr[np.nonzero(pr)])

    # compute the conditional entropy (if asked)
    # if nargout > 1 % compute the entropy
    # entropy = log(bdwidth*T(end,end)) - pr'*logp;
    # end

    
    # Compute the conditional score

    tmp = np.array([logp[i].toarray() for i in ix[nr,:].astype(int)]).reshape(kerp.shape)
    psi = np.sum(tmp*kerp ,axis=1)   #nr = 1:n repeated p times
    psi = psi.reshape(n, p,order='F')/bdwidth

    tmp = np.sum(psi,axis=0)/n
    psi = psi - tmp*np.ones((n,1))

    lam = np.matmul(psi.T,data)/n
    lam = np.tril(lam) + np.tril(lam,-1).T
    lam[p-1,p-1] = lam[p-1,p-1] - 1

    if q>0:
        psi = np.matmul( np.concatenate([ np.zeros((n,q)), psi-np.matmul(data,lam)],axis=1) , np.linalg.inv(T.T))
    else:
        psi = np.matmul(  psi-np.matmul(data,lam), np.linalg.inv(T.T))
    
    return psi #, entropy


# data = np.array([[1.1176 , -1.3416  , 0.8569],
#                  [-0.4671 ,   0.6121 ,  -0.5559],
#                  [-0.1958  ,  1.2665   , 1.4063],
#                  [-1.5295  , -1.0023  , -0.3429],
#                  [1.0748  ,  0.4653  , -1.3644],
#         ])
# psi = scorecond(data)
# print(psi)