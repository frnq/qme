import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.linalg import toeplitz
import math

# function to obtain the Floquet propagator U(t) in temrs of Floquet blocks
# U -> U(t) = ( ..., U[2], U[1], U[0], U[-1], U[-2], ...)
# for a harmonically driven system
# arguments
# times:    time steps
# omega:    driving frequency
# nphotons: number of photons driving the interaction
# P0:       time-independent part of the Liouville superoperator
# Pint:     superoperator of the interaction
# average:  if average is Ture, average over all possible phases of driving field.

def floquet_decoherence(times, omega, nphotons, P0, Pint, average = False):

    # following convention on superoperator
    P0 = 1j * P0
    Pint = 1j * Pint
    
    # number of photons
    if nphotons > 0:
        # to simplify things later, make n_photon odd.
        nphotons = nphotons + (nphotons + 1) % 2    # % --> modulo 

    # initial generator
    U0 = np.eye(len(P0)) 
    
    # ---> Build floquet matrix, compute matrix exponential, resum propagators and return
    # Build floquet matrix
    maxn = math.floor(nphotons / 2);
    photon_vec = np.arange(-maxn, maxn+1, 1)
    photon_diag = sp.sparse.eye(nphotons) 
    photon_diag.setdiag(photon_vec) 
    photonmatrix = omega * sp.sparse.kron(photon_diag, sp.sparse.eye(len(P0)))

    # system
    Pf = sp.sparse.kron(sp.sparse.eye(nphotons), csr_matrix(P0)) 
    # system + photons
    Pf = Pf + photonmatrix

    # add Floquet interaction term
    tempv = np.zeros((1, nphotons))
    tempv[0,1] = 1
    Pf = Pf + sp.sparse.kron(csr_matrix(toeplitz(tempv)), csr_matrix(Pint))
    # Build U0_f
    tempv2 = np.zeros((1, nphotons))
    tempv2[0,maxn] = 1
    tempv2
    U0_f = np.kron(tempv2.T, U0)
    
    # Compute matrix exp
    # Find eigenstates of the Floquet matrix
    evs_Pf, evecs_Pf = np.linalg.eig(Pf.todense())
    
    # Prepare U_t for output
    U_t = np.zeros((len(P0), len(P0), len(times)), dtype=complex)

    # 
    for tc in range(len(times)):
        U_eigs = np.diag(np.exp(-1j * times[tc] * evs_Pf))
        inv_evecs_Pf = np.linalg.inv(evecs_Pf)
        U_f = (evecs_Pf @ (U_eigs@inv_evecs_Pf)) @ U0_f

        # resum U_f to get U_t
        if average is True:
            U_t[:,:,tc] = U_t[:,:,tc] + U_f[maxn * len(P0) + np.arange(0,len(P0)), :]
        else:
            for jc in np.arange(-maxn,maxn+1):
                counter = jc + maxn
                U_t[:,:,tc] = U_t[:,:,tc] + U_f[counter * len(P0) + np.arange(0,len(P0)), :] * np.exp(1j * jc * omega * times[tc])

    return U_t
