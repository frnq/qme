import numpy as np
from numpy import kron
from scipy.linalg import eig
import matplotlib.pyplot as plt

# function to generate the Liouville superoperator
def Liouvillian(H, Ls, hbar = 1):
    d = len(H) # dimension of the system
    superH = -1j/hbar * ( np.kron(np.eye(d),H)-np.kron(H.T,np.eye(d)) ) # Hamiltonian part
    superL = sum([np.kron(L.conjugate(),L) 
                  - 1/2 * ( np.kron(np.eye(d),L.conjugate().T.dot(L)) +
                            np.kron(L.T.dot(L.conjugate()),np.eye(d)) 
                          ) for L in Ls])
    return superH + superL

# parameters
Omega = 0.05
Gamma = Omega/5

# Build the superoperator
H_s = np.array([[0, Omega], [Omega, 0]]);
L = np.sqrt(Gamma)*np.array([[0, 1], [1, 0]]);
P = Liouvillian(H_s,[L])

# Finding matrices containing normalised right and left eigenvectors:
D,left,right = eig(P, left=True)

# initial state in vectorised form
vec_rho_0 = np.array([0,0,0,1]); 

# solution
ts = np.linspace(0, 200, 50); 
vec_rho_t = np.zeros((len(vec_rho_0), len(ts)))

for k in range(len(H_s)**2):
    norm_fac = np.sqrt(left[:,k].T.conjugate().dot(right[:,k]))
    left_k_dag_norm = left[:,k].T.conjugate()/norm_fac
    right_k_norm = right[:,k]/norm_fac
    ak = left_k_dag_norm.dot(vec_rho_0)
    vec_rho_t = vec_rho_t + ak*np.array([right_k_norm*np.exp(D[k]*t) for t in ts]).T

# Populations dynamics
fig, ax = plt.subplots()
ax.plot(ts*Omega, np.real(vec_rho_t[0]), 'k.-', label = r'$\rho_{00}(t)$' );
ax.plot(ts*Omega, np.real(vec_rho_t[3]), 'r.-', label = r'$\rho_{11}(t)$' );
ax.set_xlabel(r'time ($t\Omega$)')
ax.set_ylabel(r'population')
ax.legend();
