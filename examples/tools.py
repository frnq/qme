import numpy as np
import matplotlib.pyplot as plt

# calculated sorted eigenvalues
def eigensorter(H, threshold = 1e-12):
    # calculate eigenvalues and eigenvectorss
    evals, evecs = np.linalg.eig(H)
    # get the list of sorted indices from the eigenvalues
    ids = np.argsort(evals)
    # sort the eigenvalues
    evals = evals[ids]
    # sort the eigenvectors
    evecs = evecs[:,ids]
    # transpose the vectors
    evecs = evecs.T
    # test the result
    error = np.linalg.norm([evals[k]-vec.conj().T@H@vec for k,vec in enumerate(evecs)])
    if error > threshold:
        # inform if error is too large
        print('Error size: ',error)
        return error
    else:
        # otherwise return results
        return evals,evecs

# import tqdm to monitor progress if the package is installed
def progressbar(iterator):
    # check if the package exists
    try:
        # import progress bar method
        from tqdm import tqdm
        # return progress bar iterator
        return tqdm(iterator)
    except ModuleNotFoundError:
        # otherwise leave iterator unchanged
        return iterator

# plot sphere
def plt_sphere(ax, radius = 1, color = 'black', alpha = 0.05):
        # draw sphere
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = radius*np.cos(u)*np.sin(v)
        y = radius*np.sin(u)*np.sin(v)
        z = radius*np.cos(v)
        ax.plot_surface(x, y, z, color=color, alpha=alpha)
        u = np.linspace(0,2*np.pi,100)
        x = radius*np.cos(u)*np.sin(-np.pi/2)
        y = radius*np.sin(u)*np.sin(-np.pi/2)
        z = radius*np.cos(-np.pi/2*np.ones(len(u)))
        ax.plot(x,y,z, lw = 1,color = color, alpha = 0.2);
        ax.plot(z,y,x, lw = 1,color = color, alpha = 0.2);
        R,N,N = np.linspace(-radius,radius,100),np.zeros(100),np.zeros(100)
        ax.plot(R,N,N,lw=1,c='r',alpha=0.2)
        ax.plot(N,R,N,lw=1,c='g',alpha=0.2)
        ax.plot(N,N,R,lw=1,c='b',alpha=0.2)
        return 
