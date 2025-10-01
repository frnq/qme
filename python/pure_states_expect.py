import numpy as np 

d = 3 # the system's dimension
basis = np.eye(d) # orthonormal basis (using identity matrix)
cs = np.array([1/np.sqrt(3),1j/np.sqrt(3),-1/np.sqrt(3)]) # some coefficients (normalised)
psi = sum([c*basis[j] for j,c in enumerate(cs)]) # the state psi
A = np.array([[1,0,0],[0,2,0],[0,0,3]]) # some operator
# expectation value of A in state psi
exp_A = np.real(np.conjugate(psi).T @ A @ psi)
