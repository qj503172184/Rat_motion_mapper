import numpy as np
import scipy as sp
import matplotlib

def findListKLDivergences(data,data2):
    logData = np.log(data)
    entropies = -np.sum(data*logData, axis = 1)
    logData2 = np.log(data2)  
    D = np.dot(-data, np.transpose(logData2))
    D = D - entropies[:,None]     
    D = D / np.log(2) 
    return D

def returnCorrectSigma_sparse(ds,perplexity,tol,maxNeighbors):
    s = np.shape(ds)   
    
    highGuess = np.max(ds)
    lowGuess = 1e-10

    sigma = 0.5*(highGuess + lowGuess)
    sortIdx = np.argsort(ds)
    ds = ds[sortIdx[0:maxNeighbors]]
    p = np.exp(-0.5*(ds**2)/(sigma**2))
    p = p/np.sum(p)
    idx = p>0
    H = np.sum(-p[idx]*np.log(p[idx])/np.log(2), axis = 0)
    P = 2**H
    
    if np.abs(P-perplexity) < tol:
        test = False
    else:
        test = True
    
    while test:
        
        if P > perplexity:
            highGuess = sigma
        else:
            lowGuess = sigma
    
        
        sigma = 0.5*(highGuess + lowGuess)
        
        p = np.exp(-0.5*(ds**2)/(sigma**2))
        p = p/np.sum(p)
        idx = p>0
        H = np.sum(-p[idx]*np.log(p[idx])/np.log(2), axis = 0)
        P = 2**H
        
        if np.abs(P-perplexity) < tol:
            test = False
            
    #if s[0] == 1:
    #    p = sp.sparse.coo_matrix((p,([0*maxNeighbors],sortIdx[0:maxNeighbors])),shape = (s[0],s[1]))
    #else:
    #   p = sp.sparse.coo_matrix((p,(sortIdx[0:maxNeighbors],[0*maxNeighbors])),shape = (s[0],s[1]))
        
    return sigma, p


def calculateKLCost(x,z,p):
    x0,x1 = x

    out = np.log(sum(1/(1+(x0-z[:,0])**2+(x1-z[:,1])**2))) + sum(p[p>0]*np.log(1+(x0-z[:,0])**2+(x1-z[:,1])**2))
    return out

def embedding_Z(data, training_data, motion_map, batchSize, perplexity = 32,sigmaTolerance = 1e-5,maxNeighbors = 200):
    N = data.shape[0]
    zValues = np.zeros((N,2))
    zGuesses = np.zeros((N,2))
    zCosts = np.zeros((N,1))
    batches = int(round(N/batchSize))
    inConvHull = np.zeros((N,1))
    meanMax = np.zeros((N,1))
    
    for j in range(batches):
        print('processing batch {} out of {}..........................................').format(j,batches)
        idx = np.arange(j*batchSize, (j+1)*batchSize)
        current_guesses = np.zeros((idx.shape[0],2))
        current = np.zeros((idx.shape[0],2))
        currentData = data[idx,:]
        tCosts = np.zeros(idx.shape[0])
        current_poly = np.zeros((idx.shape[0],1))
        
        D2 = findListKLDivergences(currentData,training_data)

        current_meanMax = np.zeros((idx.shape[0],1))
        
        for i in range(idx.shape[0]):
            _, p = returnCorrectSigma_sparse(D2[i,:],perplexity,sigmaTolerance,maxNeighbors)
            idx2 = np.argwhere(p>0)
            z = motion_map[idx2, :]
            z = np.reshape(z,(-1,2))
            maxidx = np.argmax(p)
            a = np.sum(z*p[p>0,None], axis = 0)
    
            guesses = np.asarray((a, motion_map[maxidx]))
            b = np.zeros((2,2))
            c = np.zeros((2,1))
            q = sp.spatial.ConvexHull(z)
            q = z[q.vertices,:]
            
            xop1 = sp.optimize.fmin(func=calculateKLCost, x0 = guesses[0,:], args=(z,p,))
            xop2 = sp.optimize.fmin(func=calculateKLCost, x0 = guesses[1,:], args=(z,p,))
            fop1 = calculateKLCost(xop1,z,p)
            fop2 = calculateKLCost(xop2,z,p)
            b[0,:] = xop1
            b[1,:] = xop2
            c[0] = fop1
            c[1] = fop2
            p = matplotlib.path.Path(q)
            polyIn = p.contains_points(b)
    
            polyIn[polyIn == True] = 1
            polyIn[polyIn == False] = 0
            
            if np.sum(polyIn) > 0:
                pp = np.argwhere(polyIn!= 0)
                mI = np.argmin(c[polyIn==1])
                mI = pp[mI]
                current_poly[i] = True
            else:
                mI = np.argmin(c)
                current_poly[i] = False

            current_guesses[i,None] = guesses[mI,:]
            current[i,:] = b[mI,:]
            tCosts[i] = c[mI]
            current_meanMax[i] = mI
        
        zGuesses[idx,:] = current_guesses
        zValues[idx,:] = current
        inConvHull[idx] = current_poly
        meanMax[idx] = current_meanMax
    
    zValues[np.argwhere(inConvHull==False)] = zGuesses[np.argwhere(inConvHull == False)]
    
    return zValues