import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

H = np.array([[0,1],[1,0.5]]) # Hamiltonian
c_ops = [np.array([[0,1],[0,0]])] # Lindblad operators

superop = Liouvillian(H,c_ops) # superoperator using QuTiP
rho0 = np.array([[1,0],[0,0]]) # initial state

# propgation using exmp from scipy
def Propagate(rho0, superop, t):
    d = len(rho0) # dimension of the system
    propagator = expm(superop * t) # propgator
    vec_rho_t = propagator @ np.reshape(rho0,(d**2,1)) # apply to initial state
    return np.reshape(vec_rho_t,(d,d)) # return rho(t) using Qobj class

# time steps
times = np.linspace(0,10,100)
# Population of the basis state with index i = 0 using:
pops = np.array([ np.real(np.trace(Propagate(rho0,superop,t)@rho0)) for t in times]) # expm

# plot
fig, ax = plt.subplots()
ax.plot(times, pops, 'b-', label = 'w/ expm');
ax.set_ylabel(r'Population $\rho_{00}(t)$')
ax.set_xlabel(r't')
ax.legend();
ax.set_aspect(7);
fig.savefig('figures/propagation.png',transparent=True, dpi = 600,bbox_inches='tight')
