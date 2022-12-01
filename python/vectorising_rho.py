import numpy as np

psi = np.array([1,2j,0,-2,0]) # some (non-normalised) state
rho = np.outer(psi,psi.conjugate()) # its density matrix
rho /= np.trace(rho) # normalised density matrix
d = len(psi) # the dimension of the sytems's space
vec_rho = np.reshape(rho, (d**2,1)) # vectorised density matrix
