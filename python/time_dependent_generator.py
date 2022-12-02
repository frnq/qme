import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

H0 = np.array([[1,0],[0,-1]]) # Hamiltonian H0
H1 = np.array([[0,1],[1,0]]) # Hamiltonian H1
c_ops = [np.array([[0,1],[0,0]])] # Lindblad operators

L0 = Liouvillian(H0,c_ops) # superoperator L0
rho0 = np.array([[1,0],[0,0]]) # initial state

# time-dependent coupling
v = lambda t: np.cos(t/2)

# propgation using exmp from scipy
def Propagate(rho0, superop, t):
    d = len(rho0) # dimension of the system
    propagator = expm(superop * t) # propgator
    vec_rho_t = propagator @ np.reshape(rho0,(d**2,1)) # apply to initial state
    return np.reshape(vec_rho_t,(d,d)) # return rho(t) 

# time steps
times = np.linspace(0,10,200)
# Population of the basis state with index i = 0 using:
pops = np.array([np.real(np.trace(Propagate(rho0, L0 + Liouvillian(v(t)*H1,[]),t)@rho0)) 
                 for t in times]) # expm

# plot
fig, ax = plt.subplots(figsize = (6,3))
ax.plot(times, pops, 'b-', label = 'w/ expm');
ax.set_ylabel(r'Population $\rho_{00}(t)$')
ax.set_xlabel(r't')
ax.legend();
fig.savefig('figures/td_propagation.png',transparent=True, dpi = 600,bbox_inches='tight')
