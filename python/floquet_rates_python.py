import numpy as np
import scipy as sp

# eigenvalue decomposition and sorting
def eigen_sorter(H):
    evals,evecs = sp.linalg.eig(H)
    _zipped = list(zip(evals, range(len(evals))))
    _zipped.sort()
    evals, perm = list(zip(*_zipped))
    evecs = np.array([evecs[:, k] for k in perm])
    return np.array(evals),np.array(evecs)

# floquet transition probabilities
def floquet(H0,Hint,omega,n_ph,measvec):
    # overlap probability
    overlap_prob = 0
    # spectral decomposition of H0
    evals_0,evecs_0 = eigen_sorter(H0)
    # atom Hamiltonian
    H_atom = np.kron(np.eye(n_ph),H0)
    # photon range
    max_ph = int(np.floor(n_ph/2))
    # dimension of the system
    dim = len(H0)
    # photon Hamiltonian
    H_ph = omega*np.kron(np.diag([k for k in range(-max_ph,max_ph+1)]),np.eye(dim))
    # interactions
    temp_v = np.array([int(k==1) for k in range(n_ph)])
    # interaction Hamiltonian
    H_int = np.kron(sp.linalg.toeplitz(temp_v),Hint)
    # construct the full Hamiltonian
    H = H_atom + H_ph + H_int
    # ground state
    psi_g = np.kron(np.array([int(k==max_ph) for k in range(n_ph)]),evecs_0[-1])
    # spectral decomposition of H
    evals,evecs = eigen_sorter(H)
    # loop over all the transitions
    for k_c in range(1,n_ph+1):
        temp_v = np.array([int(k==k_c) for k in range(n_ph)])
        psi_m = np.kron(temp_v,measvec)
        for evec in evecs:
            overlap = psi_m.T.conjugate()@np.outer(evec.T.conjugate(),evec)@psi_g
            overlap_prob += np.real(np.conjugate(overlap)*overlap)
    # return total transition probability
    return overlap_prob
