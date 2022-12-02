import numpy as np

def BR_tensor(H,a_ops,secular=True,secular_cut_off = 0.01):
    dim = len(H) # dimension
    evals,ekets = np.linalg.eig(H) # HS's basis
    # sort basis
    _zipped = list(zip(evals, range(len(evals))))
    _zipped.sort()
    evals, perm = list(zip(*_zipped))
    ekets = np.array([ekets[:, k] for k in perm])
    evals = np.array(evals)
    # coupling ops in ekets basis
    a_ops_S = [[np.linalg.inv(ekets)@a@np.linalg.inv(ekets),nps] for a,nps in a_ops] 
    # Bohr frequencies (w_ab)
    indices = [(a,b) for a in range(dim) for b in range(dim)]
    BohrF = np.sort(np.array([evals[a]-evals[b] for a in range(dim) for b in range(dim)]))
    R = np.zeros((dim**2,dim**2),dtype = complex) # construct empty R
    for j,(a,b) in enumerate(indices): # loop over indices
        for k,(c,d) in enumerate(indices): # loop over indices
            # unitary part
            R[j,k] += -1j * (a==c)*(b==d)*(evals[a]-evals[b])
            for a_op,nps in a_ops_S: # loop over uncorrelated a_ops
                gmax = np.max([NPS(f) for f in BohrF]) # largest rate for secular approximation
                A = a_op # coupling operator
                # secular approximation test
                if secular is True and abs(evals[a]-evals[b]-evals[c]+evals[d]) > gmax*secular_cut_off:
                    pass
                else:      
                    # non-unitary part
                    R[j,k] += - 1/2 * ((b==d)*np.sum([A[a,n]*A[n,c]*nps(evals[c]-evals[n])  
                                                      for n in range(dim)])
                                                     -A[a,c]*A[d,b]*nps(evals[c]-evals[a]) +
                                       (a==c)*np.sum([A[d,n]*A[n,b]*nps(evals[d]-evals[n])  
                                                      for n in range(dim)])
                                                     -A[a,c]*A[d,b]*nps(evals[d]-evals[b]))
    return R

e0, delta = 1,0.2 # spin parameters
sz,sx = np.array([[1,0],[0,-1]]), np.array([[0,1],[1,0]])
HS = e0/2 * sz + delta/2 * sx # spin Hamiltonian

def S(w,wc,eta,beta,thresh = 1e-10): # Noise Power Spectum
    return (2*np.pi*eta*w*np.exp(-abs(w)/wc) /
            (1-np.exp(-w*beta)+thresh)*(w>thresh or w<=-thresh) +
            2*np.pi*eta*beta**-1*(-thresh<w<thresh)) 

# noise power spectrum
NPS = lambda w: S(w,wc=1,eta=1,beta=2)
a_ops = [[sz, NPS]] # coupling operator and associated NPS

BR_tensor(HS,a_ops)
