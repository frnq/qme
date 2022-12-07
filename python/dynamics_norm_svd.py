import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

# parameters
Omega = 0.05
Gamma = Omega/5

# Build the superoperator
H_s = np.array([[0, Omega], [Omega, 0]]);
L = np.sqrt(Gamma)*np.array([[0, 1], [1, 0]]);
superop = Liouvillian(H_s,[L])

# Finding matrices containing normalised right and left eigenvectors:
D,left,right = eig(superop, left=True)

# initial state in vectorised form
vec_rho_0 = np.array([0,0,0,1], dtype = complex); 

# solution
ts = np.linspace(0, 200, 200); 
vec_rho_t = np.zeros((len(vec_rho_0), len(ts)))

for k in range(len(H_s)**2):
    norm_fac = np.sqrt(left[:,k].T.conjugate().dot(right[:,k]))
    left_k_dag_norm = left[:,k].T.conjugate()/norm_fac
    right_k_norm = right[:,k]/norm_fac
    ak = left_k_dag_norm.dot(vec_rho_0)
    vec_rho_t = vec_rho_t + ak*np.array([right_k_norm*np.exp(D[k]*t) for t in ts]).T

# Populations dynamics
fig, ax = plt.subplots( figsize = (6,2))
ax.plot(ts*Omega, np.real(vec_rho_t[3]), 'k-')
ax.set_xlabel(r'time ($t\Omega$)', usetex = True, fontsize = 10)
ax.set_ylabel(r'$\mathrm{Tr}[\rho(t)\rho_0]$', usetex=True, fontsize = 10);
