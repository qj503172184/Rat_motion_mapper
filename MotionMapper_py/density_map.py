import numpy as np
from matplotlib import pyplot as plt

def density_map(y, sigma, rangeVals, i):
    xx = np.linspace(rangeVals[0],rangeVals[1],i)
    yy = xx
    
    XX, YY = np.meshgrid(xx, yy)
    G = np.exp(-0.5*(np.multiply(XX,XX)+np.multiply(YY,YY))/sigma**2) / (2.*np.pi*sigma**2)
    
    
    xx =  np.append(xx,rangeVals[1])
    yy =  np.append(yy,rangeVals[1])
    
    Z , a, b= np.histogram2d(y[:,0],y[:,1], bins = (xx, yy))
    Z = Z / np.sum(Z)
    
    density = np.transpose(np.fft.fftshift(np.real(np.fft.ifft2(np.multiply(np.fft.fft2(G),np.fft.fft2(Z))))))
    density[density < 0] = 0
    return XX, YY, density

def plot_density_map(y, resolution = 1000, beta = 10):
    rangeVals = [int(-y.max()), int(y.max())]
    sigma = y.max()/beta
    XX, YY, density = density_map(y, sigma, rangeVals, resolution)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(XX, YY, density, cmap = 'CMRmap_r')
    fig.colorbar(im, ax=ax)
    plt.title('motion map')
    plt.savefig('./plots/motion_map.png')
    plt.show()
