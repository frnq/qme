import numpy as np

# constructs the Liouville superoperator from 
# the Hamilltonian and
# the set of Lindblad operators rescaled by the root of the rates
def Liouvillian(H, Ls, hbar = 1):
    d = len(H) # dimension of the system
    superH = -1j/hbar * ( np.kron(np.eye(d),H)-np.kron(H.T,np.eye(d)) ) # Hamiltonian part
    superL = sum([np.kron(L.conjugate(),L) 
                  - 1/2 * ( np.kron(np.eye(d),L.conjugate().T.dot(L)) +
                            np.kron(L.T.dot(L.conjugate()),np.eye(d)) 
                          ) for L in Ls])
    return superH + superL

H = np.array([[0,1],[1,1]]) # some Hamiltonian
Ls = [np.array([[0,1],[0,0]])] # Lindblad operators with embedded rates
superop = Liouvillian(H,Ls) # Liouville superoperator
