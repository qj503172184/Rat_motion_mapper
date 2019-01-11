import numpy as np

def Hbeta(D=np.array([]), beta=1.0):
    
    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    #print(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def compute_pjz(a = np.array([]), x=np.array([]), tol=1e-5, perplexity=30):
    # compute p value for embedding
    # a: 1*n array, wavelet array at a time point
    # x: training_batch 
    #print("Computing distances...")
    sum_x = np.sum(np.square(x), 1)
    D = np.add(np.add(-2 * np.dot(a, x.T), sum_x), np.sum(a**2))
    n = x.shape[0]
    #for i in range(n):
    #d = np.linalg.norm(a-x[i])**2
    #D[i] = d
    
    P = np.zeros((1, n))
    beta = 1
    logU = np.log(perplexity)
    
    # Compute the Gaussian kernel and entropy for the current precision
    betamin = -np.inf
    betamax = np.inf
    
    (H, thisP) = Hbeta(D, beta)
    
    # Evaluate whether the perplexity is within tolerance
    Hdiff = H - logU
    tries = 0
    while np.abs(Hdiff) > tol and tries < 50:
        
        # If not, increase or decrease precision
        if Hdiff > 0:
            betamin = beta
            if betamax == np.inf or betamax == -np.inf:
                beta = beta * 2.
            else:
                beta = (beta + betamax) / 2.
        else:
            betamax = beta
            if betamin == np.inf or betamin == -np.inf:
                beta = beta / 2.
            else:
                beta = (beta + betamin) / 2.
    
        # Recompute the values
        (H, thisP) = Hbeta(D, beta)
        Hdiff = H - logU
        tries += 1

    # Set the final row of P
    P = thisP
    
    # Return final P-matrix
    #print(P)
    #print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def compute_qjz(b = np.array([]), y=np.array([])):
    # computing q value for embedding
    # b: 1*2 array
    # y: motion_map

    sum_y = np.sum(np.square(y), 1)
    D = np.add(np.add(-2 * np.dot(b, y.T), sum_y), np.sum(b**2))
    num = 1. /(1.+D)
    # sum(Q) must be 1
    Q = num / np.sum(num)
    Q = np.maximum(Q, 1e-12)
    
    return num, Q





def embedding_Z( (index, x, training_batch, motion_map) ):
    # embedding feature vector into motion map
    # index: time idx of wavelet array
    # x: wavelet array
   
    print('Computing embedding value #{};').format(index)
    # Initialize variables
    
    max_iter = 30000
    #momenum is not necessay when embedding? (probably wrong)
    #initial_momentum = 0.5
    #final_momentum = 0.8
    eta = 50
    min_gain = 0.01
    
    iY = np.zeros((1, 2))
    gains = np.ones((1, 2))
    
    # Compute P-values
    P = compute_pjz(x[index], training_batch)
    # sum(P) must be 1
    P = P / np.sum(P)
    
    
    P = np.maximum(P, 1e-12)
    
    # initiate y value
    y = np.zeros((1,2))
    for i in range(len(P)):
        y += P[i]*motion_map[i]
    #print('{}: {}').format(index, y)

    # Run iterations
    for iter in range(max_iter):
        dY = np.zeros((1, 2))
        # Compute q value
        num, Q = compute_qjz(y, motion_map)
        # Compute gradient
        dY = np.sum(np.tile((P-Q) * num, (2, 1)).T * (y-motion_map), 0)
        # Perform the update
        #if iter < 20:
        #momentum = initial_momentum
        # else:
        #momentum = final_momentum
        
        gains = (gains + 0.5) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.5) * ((dY > 0.) == (iY > 0.))

        gains[gains < min_gain] = min_gain
        #print(iY)
        iY =  eta * (gains * dY)

        y = y + iY
    
        #print(dY)
    
        # Compute current value of cost function
        #if (iter + 1) % 100 == 0:
        #    C = np.sum(P * np.log(P / Q))
    #print("Iteration %d: error is %f" % (iter + 1, C))
    print('result for {}: {} ').format(index, y)
    # Return solution
    return y


