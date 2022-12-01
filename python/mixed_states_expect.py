import numpy as np 

d = 3 # the system's dimension
basis = np.eye(d) # orthonormal basis (using identity matrix)
ps = np.array([0.1,0.3,0.6]) # some probabilities (normalised)
rho = sum([p * np.outer(basis[j],basis[j].conjugate()) for j,p in enumerate(ps)]) # density operator
A = np.array([[1,0,0],[0,2,0],[0,0,3]]) # some operator
# expectation value of some operator in some state
exp_A = np.trace(rho @ A)
