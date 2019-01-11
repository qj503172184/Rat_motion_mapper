import numpy as np

def find_Eigen(data, cov_matrix, num):
    if cov_matrix.shape == (len(data[0]), len(data[0])):
        pass
    else:
        print('Wrong covariance matrix format')
    
    values, vecs = np.linalg.eig(cov_matrix)
    eig_values = values[:num]
    eig_vecs = vecs[:,:num]
    return eig_values, eig_vecs


def find_Covariance(data, batch_size):
    count = 0
    for i in range(len(data)//batch_size):
        # Initialize covariance matrix
        if count == False:
            temp = np.zeros((len(data[0]), batch_size))
            for j in range(batch_size):
                temp[:,j] = data[j]
            Cov_matrix = np.cov(temp)
            count += 1
        else:
            # calculate covariance batch by batch
            if i < len(data)//batch_size:
                temp = np.zeros((len(data[0]), batch_size))
                for j in range(batch_size):
                    temp[:,j] = data[j+count*batch_size]
                # add each batch
                Cov_matrix += np.cov(temp)
                count += 1
            else:
                temp = np.zeros((len(data[0]),len(data)%batch_size))
                for j in range(len(data)%batch_size):
                    temp[:,j] = data[j+count*batch_size]
                Cov_matrix += np.cov(temp)
                count += 1
    return Cov_matrix

def find_Projection(data, eigen_vecs):
    temp = eigen_vecs.transpose()
    projection_matrix = data.dot(temp)
    return projection_matrix

def fastWavelet_morlet_convolution_parallel((x, f, omega0, dt, count)):
    print("Calculating morlet wavelet transform of eigenmodes {}......").format(count)
    N = len(x)
    L = len(f)
    amp = np.zeros((L,N))
    if np.mod(N,2) == 1:
        x.append(0)
        N = N+1
        test = True
    else:
        test = False
    
    s = np.shape(x)
    if len(s) != 1:
        raise ValueError('dimension of array must be 1*N')
    
    x = [[0]*(N/2),x,[0]*(N/2)]
    x = [num for ele in x for num in ele]
    M = N
    N = len(x)
    
    scales = [(omega0 + np.sqrt(2+omega0**2))/(4*np.pi*i) for i in f]
    Omegavals = [2*np.pi*i/(N*dt) for i in range(-N/2,N/2)]
    
    xHat = np.fft.fft(x)
    xHat = np.fft.fftshift(xHat)
    
    if test:
        idx = M/2+M-1
    else:
        idx = M/2+M
    
    for i in range(L):
        
        m = [np.pi**(-1/4)*np.exp(-0.5*(-temp*scales[i]-omega0)**2) for temp in Omegavals]
        q = np.fft.ifft(np.multiply(m,xHat))
        q = [temp*np.sqrt(scales[i]) for temp in q]
        q = q[(M/2):idx]
        amp[i,:] = np.abs(q)*np.pi**(-0.25)*np.exp(0.25*(omega0-np.sqrt(omega0**2+2))**2)/np.sqrt(2*scales[i])
    print("Done eigenmode {}! ").format(count)

    return amp
