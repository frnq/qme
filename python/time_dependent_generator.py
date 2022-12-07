import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

H0 = np.array([[1,0],[0,-1]]) # Hamiltonian H0
H1 = np.array([[0,1],[1,0]]) # Hamiltonian H1
c_ops = [np.array([[0,1],[0,0]])] # Lindblad operators

L0 = Liouvillian(H0,c_ops) # superoperator L0
rho0 = np.array([[1,0],[0,0]]) # initial state

# time-dependent coupling
v = lambda t: np.cos(t)

# propgation using exmp from scipy
def Propagate(rho0, superop, t):
    d = len(rho0) # dimension of the system
    propagator = expm(superop * t) # propgator
    vec_rho_t = propagator @ np.reshape(rho0,(d**2,1)) # apply to initial state
    return np.reshape(vec_rho_t,(d,d)) # return rho(t) 

# time steps
times = np.linspace(0,10,100)
# Population dynamics of rho0
pops = np.array([np.real(np.trace(Propagate(rho0, L0 + Liouvillian(v(t)*H1,[]),t)@rho0)) 
                 for t in times]) # expm
# plot
fig, (ax,av) = plt.subplots(2,1,figsize = (6,3),gridspec_kw={'height_ratios':[2,1]})
fig.subplots_adjust(hspace=0.1)
ax.plot(times, pops, 'b.-');
ax.set_ylabel(r'$\mathrm{Tr}[\rho(t)\rho_0]$', usetex=True, fontsize = 10)
ax.set_xticks([])
av.set_xlabel(r'$t$', usetex = True, fontsize = 10)
av.plot(times,v(times),'k-');
av.set_ylabel(r'$v(t)$', usetex = True,fontsize = 10);
