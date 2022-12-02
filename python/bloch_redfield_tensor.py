import numpy as np
from qutip import *

e0, delta = 1,0.2 # spin parameters
HS = e0/2 * sigmaz() + delta/2 * sigmax() # spin Hamiltonian
dim = len(HS.full()) # dimension
evals,ekets = HS.eigenstates() # HS's basis
# Bohr frequencies (w_ab)
BohrF = np.sort(np.array([evals[a]-evals[b] for a in range(dim) for b in range(dim)]))

def S(w,wc,eta,beta,thresh = 1e-10): # Noise Power Spectum
    return (2*np.pi*eta*w*np.exp(-abs(w)/wc) /
            (1-np.exp(-w*beta)+thresh)*(w>thresh or w<=-thresh) +
            2*np.pi*eta*beta**-1*(-thresh<w<thresh)) 

# noise power spectrum
NPS = lambda w: S(w,wc=1,eta=1,beta=2)
a_ops = [[sigmaz(), NPS]] # coupling operator and associated NPS
a_ops_S = [[a.transform(ekets),nps] for a,nps in a_ops] # coupling ops in ekets basis

# secular approximation
secular = True
secular_cut_off = 0.01

R = np.zeros((dim**2,dim**2),dtype = complex) # construct empty R
for j,(a,b) in enumerate(indices): # loop over indices
    for k,(c,d) in enumerate(indices): # loop over indices
        # unitary part
        R[j,k] += -1j * (a==c)*(b==d)*(evals[a]-evals[b])
        for a_op,nps in a_ops_S: # loop over uncorrelated a_ops
            gmax = np.max([NPS(f) for f in BohrF]) # largest rate for secular approximation
            A = a_op.full() # coupling operator
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
